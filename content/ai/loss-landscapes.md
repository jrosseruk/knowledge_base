# Visualising loss landscapes

**Status:** stub — placeholder so the homepage timeline has something to link to.

## The one-liner

Random 2-D slices through parameter space are almost uninformative unless you
**filter-normalise** the basis vectors (Li et al. 2018): scale each
convolutional / linear filter of the random directions to have the same norm
as the corresponding filter in the trained model. Otherwise you're just
plotting filter-norm ratios in disguise.

## Coming soon

- A minimal PyTorch reproduction on a small CNN.
- Why these plots are suggestive but not diagnostic — same landscape, different
  basis choice, different story.
- Connections to mode connectivity and linear-mode-connectivity results.
