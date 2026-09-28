# RGB-D geometry and file contract

Use an adapter when an existing project has a different schema. The skill does
not require renaming the project's public API or replacing its coordinate system.

## Minimal interoperable fixture

`rgb.npy`: uint8 `[N,H,W,3]`, RGB order, already processed.

`predictions.npz` (load with `allow_pickle=False`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `frame_names` | `[N]` | Safe logical frame identifiers |
| `depth` | `[N,H,W,1]` | Positive camera-Z depth in scene units |
| `depth_conf` | `[N,H,W]` | Confidence; not a probability unless calibrated |
| `intrinsics` | `[N,3,3]` | Pinhole K for saved RGB/depth resolution |
| `extrinsics_world_to_camera` | `[N,3,4]` | OpenCV right/down/forward W2C |

For sensor data, zeros/NaNs may represent missing depth: mask them rather than
requiring all pixels to be positive. Check shapes, finite valid values, positive
focal lengths, rotation orthogonality/determinant and nonsingular transforms.

## Transforms

For column vectors, `p_camera = R_wc @ p_world + t_wc`.
Embed W2C in a homogeneous 4×4 matrix `E`; log `C = inverse(E)` as the camera's
pose relative to the world. `C[:3,3]` is the camera center; W2C translation is not.

For pixel `(u,v)` and Z-depth `z`:

```text
p_camera = z * inverse(K) @ [u, v, 1]
p_world  = C[:3,:3] @ p_camera + C[:3,3]
```

For NumPy row-vector point arrays this becomes
`world = camera @ C[:3,:3].T + C[:3,3]`.
Euclidean ray length must first be converted to Z-depth. Distorted imagery must
be undistorted or handled by the matching camera model before pinhole projection.

For resize then crop, adjust `fx,fy,cx,cy` consistently with image sampling.
The exact pixel-center convention matters; reuse the producer's transformed K
rather than guessing a scale from the original image dimensions.

## Rerun entity hierarchy

A common arrangement (verify types against the installed SDK):

```text
world
  cloud                 Points3D in world coordinates
  camera                Transform3D: camera-to-world, time varying
    image               Pinhole: K, processed resolution, RDF axes
      rgb               Image
      depth             DepthImage
```

Set `frame` once per image, and log its pose, pinhole, RGB and depth at that time.
A static fused cloud may be logged once. Individual historical camera frustums
can be static, while `world/camera` represents only the currently selected frame.

`Spatial3DView(... eye_controls=EyeControls3D(tracking_entity="/world/camera/image"))`
tracks the pinhole camera. Keep `world/cloud` and the active pinhole in the view's
contents, but omit its RGB/depth children if textured image planes or automatic
depth backprojection obscure the intended cloud rendering.

Numerically test known nonidentity rotations and translations, not only identity
poses. Reproject transformed points and check expected pixels on synthetic data.
An inverse/roundtrip check can pass when both operations share the same wrong
convention; analytic scene surfaces and visual evidence are additional checks.
