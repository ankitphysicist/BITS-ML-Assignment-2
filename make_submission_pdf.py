"""Create a submission PDF using only verified user-provided links and screenshot."""
import argparse
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", default="")
    parser.add_argument("--app", default="")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--output", default="submission_draft.pdf")
    args = parser.parse_args()
    styles = getSampleStyleSheet()
    story = [Paragraph("Machine Learning - Assignment 2", styles["Title"]), Spacer(1, 18)]
    ready = args.github.startswith("https://") and args.app.startswith("https://") and args.screenshot and args.screenshot.exists()
    if not ready:
        story.append(Paragraph("INCOMPLETE - NOT READY FOR SUBMISSION", styles["Heading2"]))
    for number, label, url in [(1,"GitHub repository",args.github),(2,"Streamlit application",args.app)]:
        story.append(Paragraph(f"{number}. {label}", styles["Heading2"]))
        if url.startswith("https://"):
            safe = escape(url)
            story.append(Paragraph(f'<link href="{safe}">{safe}</link>', styles["BodyText"]))
        else:
            story.append(Paragraph("[ADD VERIFIED LINK]", styles["BodyText"]))
    story.append(Paragraph("3. BITS Virtual Lab execution screenshot", styles["Heading2"]))
    if args.screenshot and args.screenshot.exists():
        img = Image(str(args.screenshot))
        w, h = img.imageWidth, img.imageHeight
        scale = min(470/w, 260/h)
        img.drawWidth, img.drawHeight = w*scale, h*scale
        story.append(img)
    else:
        story.append(Paragraph("[INSERT ACTUAL BITS LAB SCREENSHOT]", styles["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("4. README.md (full content)", styles["Title"]))
    readme = (ROOT/"README.md").read_text(encoding="utf-8")
    lines = []
    for line in readme.splitlines():
        lines.extend(textwrap.wrap(line, 90, break_long_words=True) or [""])
    for i in range(0, len(lines), 35):
        story.append(Preformatted("\n".join(lines[i:i+35]), styles["Code"], maxLineLength=92))
    doc = SimpleDocTemplate(args.output, pagesize=A4, leftMargin=45, rightMargin=45, topMargin=45, bottomMargin=45)
    doc.build(story)
    print("Created", args.output, "(complete)" if ready else "(INCOMPLETE)")

if __name__ == "__main__":
    main()
