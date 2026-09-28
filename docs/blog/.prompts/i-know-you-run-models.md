# Header image: "I know you run models" — image-generation prompt

The SVG next to this file is a placeholder sketch. This prompt is for the
final image (any capable image model), so it reads as a real poster and
carries the joke.

## The idea to get across

A parody of the famous 1917 "I want YOU" recruiting poster. The person is not
a soldier-uncle: it is a tired, slightly smug ML engineer who knows exactly
what you did last night — rented a GPU at 2 a.m. and watched vLLM crash with
"out of memory". The look says "I know. I've been there. Come here." The
feeling is a knowing wink, not an ad: funny to anyone who has fought an OOM,
still clean enough for a technical blog and LinkedIn.

## Prompt

> Vintage 1917 American recruiting-poster parody, painted lithograph style,
> warm off-white paper with slight grain and faded print edges. Waist-up
> portrait of a tired but confident software engineer in their thirties,
> wearing a dark navy hoodie with the hood up, stubble, slightly raised
> eyebrow, knowing half-smile, looking straight at the viewer. Their right arm
> points directly at the viewer, index finger foreshortened toward the camera,
> exactly the classic poster gesture. On the front of the hood, where the
> famous top hat would be, a small red warning light glows with the letters
> "OOM", casting a soft red rim light on the hood. Behind them, faint and
> out of focus, a server rack with a single blinking amber LED and a wall
> clock showing 2:00. Large bold serif lettering on the right in poster
> style: "I KNOW YOU RUN MODELS", with "MODELS" in red. Smaller italic line
> below: "On rented GPUs. At 2 a.m." A dark navy banner at the bottom:
> "APRON NEEDS YOU". Colour palette: navy, poster red, cream. Composition
> 1200x630 (link-preview ratio), figure on the left third, text on the right,
> generous margins. Clean, legible typography; no other text.

## Negative prompt / constraints

- No real person, no likeness of Uncle Sam or any real person, no flags or
  national symbols, no military uniform.
- No company logos (no NVIDIA, RunPod, vLLM, Hugging Face marks) — Apron's
  own name only in the bottom banner.
- No garbled or extra text; spell exactly: "I KNOW YOU RUN MODELS",
  "On rented GPUs. At 2 a.m.", "APRON NEEDS YOU".
- Not photorealistic; stay painted poster, so it reads as parody, not an ad.

## Alt text (for the page)

"Poster parody: an engineer in a hoodie with a red OOM light on the hood
points at you. Text: I know you run models. On rented GPUs. At 2 a.m.
Apron needs you."
