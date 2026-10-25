# Third-party notices

- **OpenAI CLIP**: https://github.com/openai/CLIP. An external dependency under the MIT license. No CLIP checkpoints or tokenizer vocabulary are redistributed in this repository. The installation file pins a source revision. Model and dataset terms remain separate.
- **AugMix**: https://github.com/google-research/augmix. `third_party/augmix_ops.py` retains its original copyright header. The Apache License 2.0 is included in `third_party/licenses/Apache-2.0-AugMix.txt`. The augmentation schedule follows the evaluated implementation: one center crop and 63 augmented views, selecting six by prediction entropy.
- **PTA**: https://github.com/hzhxmu/PTA. The shared prototype update follows the published mechanism. The optional data-manifest adapter calls a separately obtained checkout's dataset loaders. The PTA repository, its data splits and its point-cloud code are not redistributed here.

Third-party components retain their respective licenses. No license has been specified for the original FGM code.
