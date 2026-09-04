"""
Reusable helper to bundle matplotlib figures (already saved as image files)
into a single, browsable HTML report.

Any EDA script in lesson5 can reuse it after saving its plots with
`plt.savefig(...)` into a given directory:

    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.html_report import build_html_report

    # ... generate and plt.savefig(...) each plot into OUTPUT_DIR ...

    build_html_report(OUTPUT_DIR, title="My Dataset EDA")

By default every image found in the output directory is included, ordered by
creation time (i.e. the order the plots were generated), so no manual
bookkeeping of filenames is required in the calling script.
"""

import glob
import html
import os

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg")


def _discover_images(output_dir):
    paths = [
        path
        for path in glob.glob(os.path.join(output_dir, "*"))
        if path.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(paths, key=os.path.getmtime)


def _caption_from_filename(filename):
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").replace("-", " ").title()


def build_html_report(
    output_dir,
    image_filenames=None,
    title="Relatorio de Analise Exploratoria",
    report_filename="report.html",
):
    """
    Writes an HTML file into `output_dir` displaying every plot as a card
    with a heading derived from its filename.

    Parameters
    ----------
    output_dir: str
        Directory where the plots were saved (via plt.savefig) and where the
        resulting HTML report will be written.
    image_filenames: list[str] | None
        Optional explicit, ordered list of filenames (relative to
        `output_dir`) to include. If omitted, every image file found in
        `output_dir` is used, ordered by creation time.
    title: str
        Page heading and <title>.
    report_filename: str
        Name of the generated HTML file.

    Returns
    -------
    str | None
        The path to the generated HTML file, or None if no images were found.
    """
    if image_filenames is None:
        image_filenames = [os.path.basename(path) for path in _discover_images(output_dir)]

    if not image_filenames:
        print(f"Nenhuma imagem encontrada em {output_dir}; relatorio HTML nao foi gerado.")
        return None

    cards = []
    for filename in image_filenames:
        caption = _caption_from_filename(filename)
        cards.append(f"""    <section class="card">
      <h2>{html.escape(caption)}</h2>
      <img src="{html.escape(filename)}" alt="{html.escape(caption)}">
    </section>""")

    page = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; background: #fafafa; color: #222; }}
    h1 {{ text-align: center; }}
    .card {{ margin-bottom: 3rem; text-align: center; }}
    .card h2 {{ font-size: 1.05rem; color: #444; font-weight: 600; }}
    .card img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1); }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
{os.linesep.join(cards)}
</body>
</html>
"""

    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(page)

    print(f"Relatorio HTML salvo em {report_path}")
    return report_path
