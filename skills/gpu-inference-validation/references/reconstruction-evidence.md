# Evidence for RGB-D / camera reconstruction

Choose checks that can detect the plausible remaining failures. Do not impose
one universal error threshold across models, resolutions or scene types.

## Artifact and numerical checks

- Decode the saved output and compare frame counts/order to the input manifest.
- Check depth/confidence/pose/K shapes, finite valid values and declared units.
- Treat zero/missing sensor depth according to its validity mask.
- Verify positive focal lengths, proper rotations and camera-axis conventions.
- Check whether poses are W2C or C2W and whether scale/first-view alignment is used.
- Record cropped/resized image geometry; original RGB may not match predicted K.

These checks reject corrupt or collapsed outputs. They do not establish geometry
accuracy merely because every value is finite.

## Visual evidence

Inspect representative RGB/depth pairs and fused or per-frame point clouds.
Compare camera positions with changes visible in the input views. Discontinuous
input viewpoints can legitimately produce a discontinuous polyline: do not smooth
or reorder poses simply to improve appearance. Inspect occlusions, thin structures,
scene boundaries and low-confidence regions where failure is more likely.

Name point subsampling, confidence/edge filtering and visualization clipping.
A filtered rendering is not proof that all predicted pixels are reliable. Do not
interpret a model's confidence as a calibrated probability without evidence.

## Independent image correspondences

A useful consistency check obtains correspondences from RGB features, filters
obvious false matches without using model predictions, then reprojects the source
depth using the model's relative pose and target K. Report the pixel residuals,
sample count, evaluated pairs and exclusions. If scale or pose was fitted to those
same correspondences, state that explicitly; it changes what the check proves.

Pairs without sufficient overlap or reliable matches remain unverified, not
successful. Pixel errors from different input resolutions or different match sets
are not a controlled model ranking. Ground-truth depth/poses are stronger evidence
when available and applicable.

## Internal and cross-model consistency

Project one depth map into a neighboring view and compare depths where projection
is valid. Occlusion and disocclusion affect residuals: document any masks and avoid
filtering by residual then presenting the result as an unbiased accuracy measure.

Camera-center agreement between models can be compared after one declared global
Sim(3) when scale is arbitrary. Report the normalization used for relative error.
Agreement between two predictions, or a reprojection/unprojection roundtrip, is
not a substitute for ground truth.

## Performance reporting

Synchronize asynchronous device work around timing. Separate checkpoint load,
preprocessing, forward, postprocessing and disk output when those distinctions
matter. Identify warmup/compile state, precision, resolution, number of frames,
hardware, concurrency and allocated versus reserved peak memory. Do not compare
unmatched configurations as a fair speed or quality benchmark.
