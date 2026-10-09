# Demo assets

- `CP01_Overview.pdf`: two-page US Letter project overview explaining what the assistant answers, with concrete policy-question examples, a simple architecture diagram and a large real application screenshot.
- `CP01_Demo_Walkthrough.mp4`: approximately 60 seconds, 1920 x 1080, H.264. This is a silent, captioned walkthrough assembled from actual application screenshots, not a continuous screen recording.
- `01_home.jpg` through `06_workflow.jpg`: unaltered browser viewport captures used by the video.
- `answer_panel.jpg` and `trace_panel.jpg`: retained rectangular crops from the original overview. The revised PDF enlarges the answer panel directly from `02_answer.jpg`. No app text, results, citations, or trace values were reconstructed.

## What is demonstrated

The recorded run uses **Extractive demo**, the deterministic local baseline. The video shows the equipment-allowance question, the evidence desk, the actual LangGraph execution trace, an unrelated cafeteria-menu question that reaches the unknown-response branch, and the app's prototype limitations. It does not demonstrate live hosted-model API calls.

The captured helper text uses "Ollama" for the compatible API transport. The final app labels that option "Hosted LLM API" to clarify that the course-provided open-source model can run on a server. No local chat-model installation is required or shown. The screenshots' policy answers and extractive trace are unchanged by this labeling clarification. `captured_source_sha256.txt` identifies the actual captured source snapshot.

The shown policies are original fictional Northstar Labs documents. The unknown-response example establishes only the behavior of that specific question. It is not evidence of reliable general unknown detection. Exact-source checks establish provenance, not semantic correctness, applicability, or completeness. See the project evaluation and QA documentation for measured limitations.

## Rebuild the documents and video

The helper requires Python packages `reportlab`, `Pillow`, and `pypdf`, plus the `ffmpeg` executable. These are optional asset-production tools, not runtime requirements for the assistant.

```bash
python docs/assets/build_demo_assets.py
```

Use `--pdf-only` or `--video-only` to rebuild one artifact. A `--repo-url` value may be included only after the repository has actually been created and verified. Without it, the PDF refers readers to the included project files.

The script consumes the existing screenshots; it does not fabricate them. To refresh the captures, run the real Streamlit app with the supplied corpus, select **Extractive demo**, ask the shown question, and capture the visible result, expanded sources, and activity trace. Update the homepage and workflow captures whenever provider labels or implementation details change. Never include credentials or authentication screens.

Intermediate video frames and PDF renders are written to `build/`. They are not final deliverables. `video_metadata.json` records the encoded video's dimensions and duration.
