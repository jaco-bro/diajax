# Dia-JAX

**An experimental JAX port of Dia, the 1.6B parameter text-to-speech model from Nari Labs**

Dia-JAX is a work-in-progress port of the original PyTorch-based Dia model to JAX via Flax NNX.

## Features

Just like the original Dia model, Dia-JAX aims to offer:

- Generate dialogue via `[S1]` and `[S2]` tags
- Generate non-verbal elements like `(laughs)`, `(coughs)`, etc.
  - Supported verbal tags: `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning with reference audio (TODO: currently not implemented)
- Quality comparable to commercial solutions like ElevenLabs Studio 

## Quickstart

### Install via pip

```bash
pip install diajax
```

## ⚙️ Usage

**Note: Currently only recommended for experimental/development use due to memory issues**

### Run from Command Line

```bash
# Generate audio with default settings
dia --text "[S1] Dear Jacks, to generate audio from text from any machine. (applause) [S2] Really? How! (screams) [S1] With flakes and an axe. (chuckles)"

# Or with custom parameters
dia --temperature 0.7 --cfg-filter-top-k 42
```
### As a Python Library

```python
import diajax
model, config = diajax.load()
output = diajax.generate(model, config, text)

import soundfile as sf
sf.write('dia.mp3', output, 44100)
```

## Acknowledgments

This project is a port of the [original Dia model](https://github.com/nari-labs/dia) by Nari Labs. We thank them for releasing their model and code, which made this port possible.

## License

This project is licensed under the same terms as the original Dia model. See [LICENSE](LICENSE) for details.
