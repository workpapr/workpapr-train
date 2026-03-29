# @workpapr/train

Training framework for Workpapr — learn from auditor overrides to improve AI-assisted reviews.

## What it does

- **Trace recording** — captures auditor decisions during reviews (implements core's TraceHook interface)
- **MLX fine-tuning** — local LoRA training on Apple Silicon via MLX
- **SFT/DPO data conversion** — converts traces to supervised fine-tuning and direct preference optimization formats

## Install

```bash
npm install @workpapr/train
```

Requires [@workpapr/core](https://github.com/workpapr/workpapr) as a peer dependency.

## Related packages

| Package | Description |
|---------|-------------|
| [@workpapr/core](https://github.com/workpapr/workpapr) | Core audit CLI platform |
| [@workpapr/demo](https://github.com/workpapr/workpapr-demo) | Interactive flywheel demo |

## License

MIT
