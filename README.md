# Dia-JAX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaco-bro/diajax/blob/main/assets/diajax_TPU.ipynb)

**An experimental JAX port of Dia, the 1.6B parameter text-to-speech model from Nari Labs**

## Quickstart

[output.mp3](https://raw.githubusercontent.com/jaco-bro/diajax/main/assets/example_output.mp3)

```bash
pip install -U diajax
dia --text "[S1] Dear Jacks, to generate audio from text from any machine. [S2] Any machine? (gasps) How? [S1] With flakes and an axe. (chuckle) " --max-tokens=600
```

```python
import diajax
model = diajax.load()
output = diajax.generate(model, "[S1] Dear Jacks, to generate audio from text from any machine. [S2] Any machine? (laughs) How? [S1] With flacks and an axe. (coughs)")
diajax.save(output)
```

## Acknowledgments

This project is a port of the [original Dia model](https://github.com/nari-labs/dia) by Nari Labs. We thank them for releasing their model and code, which made this port possible.

## License

This project is licensed under the same terms as the original Dia model. See [LICENSE](LICENSE) for details.
