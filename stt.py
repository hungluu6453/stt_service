import torch
import torchaudio
import time
from speechbrain.pretrained import EncoderASR
from pyctcdecode import build_ctcdecoder


class Speech_to_Text:
    def __init__(self,  
                lm_file = 'models/lm/lm.binary',
                vocab_file = 'models/lm/vocab-260000.txt',
                model_path="models/asr/wav2vec2-sb",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        self.lm_file = lm_file
        self.vocab_file = vocab_file
        self.model = EncoderASR.from_hparams(source=model_path,
                                  run_opts={"device": device}
                                  )
        self.ngram_lm_model = self.get_decoder_ngram_model(self.model.tokenizer, self.lm_file, self.vocab_file)

    def get_decoder_ngram_model(self, tokenizer, ngram_lm_path, vocab_path=None):
        unigrams = None
        if vocab_path is not None:
            unigrams = []
            with open(vocab_path, encoding='utf-8') as f:
                for line in f:
                    unigrams.append(line.strip())

        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab]
        vocab_list = vocab

        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        decoder = build_ctcdecoder(vocab_list, ngram_lm_path, unigrams=unigrams)
        return decoder

    def transcribe_file(self, path, use_lm=True):
        start_time = time.time()
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.squeeze()
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                sample_rate, 16000,
            )(waveform)
        wavs = waveform.unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        with torch.no_grad():
            wav_lens = wav_lens.to("cpu")
            logits = self.model.encode_batch(wavs, wav_lens)
            predictions = self.model.decoding_function(logits, wav_lens)
            text_batch = [
                self.model.tokenizer.decode_ids(token_seq)
                for token_seq in predictions
            ]

        if use_lm:
            text_batch = [self.ngram_lm_model.decode(logits.detach().cpu().numpy()[0], beam_width=100)]
        return text_batch[0], time.time() - start_time


if __name__ == "__main__":
    stt_model = Speech_to_Text()
    file_path = "example.mp3"
    text, execution_time = stt_model.transcribe_file(file_path)
    print(text, execution_time)
