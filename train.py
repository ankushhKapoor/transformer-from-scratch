import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
from sacrebleu import corpus_bleu
from pathlib import Path
import warnings

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, alpha=0.6):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output (reuse for every step)
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the sos token
    # (batch, seq_len)
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Candidates list: (sequence, score, finished_flag)
    # We start with a log-score of 0.0
    candidates = [(decoder_initial_input, 0.0, False)]

    for _ in range(max_len):
        # If all candidates are finished, stop
        if all(cand[2] for cand in candidates):
            break

        new_candidates = []

        for candidate, score, finished in candidates:
            if finished:
                new_candidates.append((candidate, score, True))
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            
            # Decode and get probabilities for the LAST token only
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            prob = model.project(out[:, -1]) 
            
            # Get the top k probabilities and indices
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)

            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                
                new_candidate = torch.cat([candidate, token], dim=1)
                new_score = score + token_prob
                is_done = (token.item() == eos_idx)
                
                new_candidates.append((new_candidate, new_score, is_done))

        # Length Penalty calculation: score / (length ^ alpha)
        # This prevents the model from favoring extremely short sentences.
        def score_fn(cand):
            seq, score, _ = cand
            penalty = ((5 + seq.size(1)) ** alpha) / ((5 + 1) ** alpha)
            return score / penalty

        # Sort by penalized score and keep top k
        candidates = sorted(new_candidates, key=score_fn, reverse=True)[:beam_size]

    # Return the best candidate sequence
    return candidates[0][0].squeeze()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step=0,
    writer=None,
    num_examples=10
):
    model.eval()
    count = 0

    expected = []
    predicted = []

    # Accuracy counters
    total_tokens = 0
    correct_tokens = 0

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out_greedy = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, max_len, device
            )
            model_out_beam = beam_search_decode(
                model, 4, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            pred_text = tokenizer_tgt.decode(
                model_out_beam.detach().cpu().numpy()
            )

            greedy_text = tokenizer_tgt.decode(
                model_out_greedy.detach().cpu().numpy()
            )

            expected.append(target_text)
            predicted.append(pred_text)

            # ---- TOKEN-LEVEL ACCURACY ----
            target_tokens = target_text.split()
            pred_tokens = pred_text.split()

            min_len = min(len(target_tokens), len(pred_tokens))
            for i in range(min_len):
                if target_tokens[i] == pred_tokens[i]:
                    correct_tokens += 1
                total_tokens += 1

            # Count remaining unmatched tokens as incorrect
            total_tokens += abs(len(target_tokens) - len(pred_tokens))

            # Print samples
            print_msg('-' * console_width)
            print_msg(f"{'SOURCE: ':>20}{source_text}")
            print_msg(f"{'TARGET: ':>20}{target_text}")
            print_msg(f"{'PREDICTED GREEDY: ':>20}{greedy_text}")
            print_msg(f"{'PREDICTED BEAM: ':>20}{pred_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    # Compute accuracy (%)
    accuracy = 0.0
    if total_tokens > 0:
        accuracy = (correct_tokens / total_tokens) * 100.0
        print_msg(f"{'ACCURACY: '}{accuracy}%")

    bleu = 0.0

    if writer:
        # Character Error Rate
        cer_metric = torchmetrics.CharErrorRate()
        cer = cer_metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)

        # Word Error Rate
        wer_metric = torchmetrics.WordErrorRate()
        wer = wer_metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)

        # BLEU (SacreBLEU)
        bleu = corpus_bleu(predicted, [expected]).score
        writer.add_scalar("validation BLEU", bleu, global_step)

        # Token-level Accuracy
        writer.add_scalar("validation accuracy (%)", accuracy, global_step)

        writer.flush()

    return bleu, accuracy

    
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # Replace unkown token with [UNK]
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # For a word to appear in our tokenizer it should have minimum frequency of 2
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Create a fixed 90/10 split for training vs unseen validation data.
    # This ensures we measure true generalization rather than memorization.
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sequence: {max_len_src}')
    print(f'Max length of target sequence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def get_lr_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        # Avoid division by zero at step 0
        step = max(step, 1)

        # This formula (mentioned in ppr) calculates the scaling factor based on d_model and current step:
        # 1. (d_model ** -0.5) scales the LR relative to the model dimension.
        # 2. min(...) chooses between the linear warmup and the square root decay.
        #    - Linear warmup: step * (warmup_steps ** -1.5)
        #    - Inverse square root decay: step ** -0.5
        return (d_model ** -0.5) * min(
            step ** -0.5,
            step * warmup_steps ** -1.5
        )
    # LambdaLR applies the calculated factor to the lr defined in the optimizer
    return LambdaLR(optimizer, lr_lambda)

def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard - visualization and monitoring tool
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, d_model=config['d_model'])

    initial_epoch = 0
    global_step = 0
    best_bleu = float("-inf")
    # If the user specified a model to preload before training, load it
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        global_step = state['global_step']
        bleu = state['bleu']
        best_bleu = state.get('best_bleu') 

    # Simple Cross Entropy Loss fn
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1) # Apply label smoothing to avoid overconfident predictions by assigning a small probability to other classes (non top classes) and imporove generalization

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')
        model.train()
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len) -- only mask [PAD]
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len) -- also mask subsequent mask

            # Run the tensors through the transformer (encoder, decoder and the projection layer)
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # Compute the loss using a simple cross entropy
            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

        bleu = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        if bleu > best_bleu:
            best_bleu = bleu

        save_contents = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'bleu': bleu,
                'best_bleu': best_bleu
            }

        # Save the model at the end of save_every epoch number
        if config["save_every"] is not None:
            if (epoch+1) % config["save_every"] == 0:
                model_filename = get_weights_file_path(config, f'{epoch}')
                torch.save(save_contents, model_filename)

        # Save the model at the end of best bleu
        if config["save_best_only"]:
            if bleu == best_bleu:
                best_model_path = Path(config["model_folder"]) / f"{config['model_basename']}best.pt"
                torch.save(save_contents, best_model_path)

        # Save the model at the end of every epoch
        if not config["save_best_only"] and config["save_every"] is None:
            model_filename = get_weights_file_path(config, f'{epoch}')
            torch.save(save_contents, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)