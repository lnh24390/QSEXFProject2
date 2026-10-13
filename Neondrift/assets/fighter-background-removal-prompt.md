# 전투기 배경 제거

- 도구: 내장 image_gen
- 입력: 전투기.png
- 저장: 전투기-배경제거.png (원본 보관)

## 최종 프롬프트

Use case: background-extraction. Edit the attached 전투기.png. Primary request: remove only the surrounding white background and save the fighter jet as a genuine transparent-alpha PNG cutout for a game sprite. Preserve the exact top-down fighter design, silhouette, proportions, straight-up orientation, flat medium-blue wings, dark gray outline, dark oval cockpit, dark tail fins, and opaque white central fuselage. Keep white inside the fuselage opaque; make the empty background around and between wing appendages transparent. No redesign, no 3D rendering, no new gradients, no changed colors, no extra objects, no added text. Preserve the aircraft's full extent without cropping, centered on a transparent square canvas with a small transparent margin. The background must actually have alpha zero, not a white background or a rendered checkerboard.
