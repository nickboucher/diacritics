# Diacritics
*When Vision Fails: Text Attacks Against ViT and OCR*

While text-based machine learning models that operate on visual inputs of rendered text have been shown to gain robustness against a wide range of existing attacks, we demonstrate that these models become vulnerable to visual adversarial examples encoded as text. We leverage the combining diacritical mark functionality of Unicode to manipulate encoded text such that small visual perturbations appear when the text is rendered. We then propose a genetic algorithm can be used to generate such visual adversarial examples in a black-box setting, and through a user study show that the model-fooling adversarial examples do not affect human comprehension of text. We then demonstrate the effectiveness of such attacks in the real world by creating adversarial examples against production models published by Facebook, Microsoft, and IBM.

Additional information can be found in the related paper.

## Citation

If you use anything in this repository or in the *When Vision Fails* paper in your own work, please cite the following:

```bibtex
@article{boucher_visionfails_2022,
    title = {When {Vision} {Fails}: {Text} {Attacks} {Against} {ViT} and {OCR}},
    author = {Nicholas Boucher and Jenny Blessing and Ilia Shumailov and Ross Anderson and Nicolas Papernot},
    year = {2022}
}
```