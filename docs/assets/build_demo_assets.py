"""Build the two-page PDF and captioned screenshot walkthrough from real captures.

Requires reportlab, pillow, pypdf, and ffmpeg. Run from anywhere. Captures are
intentionally supplied rather than generated or reconstructed by this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
REG = FONT_DIR / 'DejaVuSans.ttf'
BOLD = FONT_DIR / 'DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('Demo', str(REG)))
pdfmetrics.registerFont(TTFont('DemoBold', str(BOLD)))
pdfmetrics.registerFontFamily('Demo', normal='Demo', bold='DemoBold',
                              italic='Demo', boldItalic='DemoBold')
INK = '#15243c'
MUTED = '#59697d'
BLUE = '#2258a8'
TEAL = '#177765'


def paragraph(c, text, x, y, width, size=9, leading=13, color=INK, bold=False):
    style = ParagraphStyle('body', fontName='DemoBold' if bold else 'Demo',
                           fontSize=size, leading=leading, textColor=HexColor(color))
    p = Paragraph(text, style)
    _, height = p.wrap(width, 1000)
    p.drawOn(c, x, y - height)
    return y - height


def screenshot(c, path, x, top, width, max_height):
    im = Image.open(path)
    area_width = width
    height = width * im.height / im.width
    if height > max_height:
        width *= max_height / height
        height = max_height
    c.drawImage(ImageReader(im), x + (area_width - width) / 2, top - height,
                width=width, height=height)
    return top - height


def page_footer(c, page):
    paragraph(c, 'CP01  |  Project overview', 36, 30, 400, 8, 11, MUTED)
    c.setFont('Demo', 8)
    c.setFillColor(HexColor(MUTED))
    c.drawRightString(576, 20, f'{page} / 2')


def arrow(c, x1, x2, y):
    c.setStrokeColor(HexColor('#7b92b0'))
    c.setFillColor(HexColor('#7b92b0'))
    c.setLineWidth(1.1)
    c.line(x1, y, x2 - 3, y)
    p = c.beginPath()
    p.moveTo(x2, y)
    p.lineTo(x2 - 4, y + 2.6)
    p.lineTo(x2 - 4, y - 2.6)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def architecture(c, top):
    """A deliberately small data-flow diagram; all labels remain real PDF text."""
    nodes = [
        ('Ask', 'Streamlit UI', 'Employee question'),
        ('Plan and retrieve', 'LangGraph', 'Coordinates roles'),
        ('Search policies', 'Chroma + MiniLM', 'Local cached index'),
        ('Select and review', 'Hosted open-source', 'model API'),
        ('Show excerpts', 'Source checks', 'Citations + trace'),
    ]
    width, gap, height = 99.2, 11, 64
    for i, (title, line1, line2) in enumerate(nodes):
        x = 36 + i * (width + gap)
        c.setFillColor(HexColor('#f0f5fb' if i != 3 else '#eaf4f0'))
        c.setStrokeColor(HexColor('#cad7e8' if i != 3 else '#b8d6c8'))
        c.setLineWidth(.7)
        c.roundRect(x, top - height, width, height, 6, fill=1, stroke=1)
        paragraph(c, title, x + 7, top - 9, width - 14, 8.35, 10.8, INK, True)
        paragraph(c, line1 + '<br/>' + line2, x + 7, top - 28,
                  width - 14, 7.9, 11, MUTED)
        if i < len(nodes) - 1:
            arrow(c, x + width + 1.5, x + width + gap - 1.5, top - height / 2)
    return top - height


def make_pdf(repo_url=None):
    """Explain the project in two Letter pages, then show a readable real answer."""
    output = ROOT / 'CP01_Overview.pdf'
    c = canvas.Canvas(str(output), pagesize=(612, 792))
    c.setTitle('CP01 Enterprise Knowledge Assistant')
    c.setAuthor('CP01 project')
    c.setSubject('Project purpose, policy questions, architecture and real application demo')
    c.setKeywords('CP01, enterprise knowledge, policy assistant, LangGraph, Chroma, demonstration')

    # Page 1: explain the outcome before the implementation.
    y = paragraph(c, 'CP01', 36, 757, 540, 10, 14, BLUE, True) - 4
    y = paragraph(c, 'Enterprise Knowledge Assistant', 36, y, 540,
                  22, 28, '#000000', True) - 10
    y = paragraph(c,
                  'An assistant for employees who need answers from company policies. '
                  'Ask a workplace question and get relevant policy excerpts with source citations. '
                  'The app shows the supporting document and the steps used to find the answer.',
                  36, y, 540, 10.4, 15) - 17
    y = paragraph(c, 'Questions it can help answer', 36, y, 540,
                  13, 18, INK, True) - 8
    paragraph(c, 'EXAMPLE QUESTION', 47, y, 253, 8, 11, MUTED, True)
    paragraph(c, 'WHAT THE POLICY ANSWER COVERS', 312, y, 253, 8, 11, MUTED, True)
    y -= 20
    examples = [
        ('Leave', 'How much annual leave do full-time staff get?',
         '24 working days a year; up to 5 carryover days expire on 31 March of the next year.'),
        ('Travel', 'What can I claim for a business trip?',
         'Hotels up to USD 180 per night before tax; meals up to USD 60 per day including tax and tips.'),
        ('Remote work', 'How often can I work from home?',
         'Eligible employees may work remotely up to 3 days a week with line-manager agreement.'),
        ('Onboarding', 'When can I claim the home-office allowance?',
         'One-time USD 300 after 30 calendar days, with approval before purchase and itemized receipts.'),
        ('Security', 'What do I need for secure remote access?',
         'A company-managed laptop, multi-factor authentication and a VPN for internal-only systems.'),
    ]
    for topic, question, answer in examples:
        c.setFillColor(HexColor('#f4f7fb'))
        c.roundRect(36, y - 45, 540, 45, 5, fill=1, stroke=0)
        paragraph(c, topic, 47, y - 5, 250, 8.5, 11.5, BLUE, True)
        paragraph(c, question, 47, y - 18, 250, 9, 12, INK)
        paragraph(c, answer, 312, y - 7, 250, 9, 12.5, INK)
        y -= 51
    y = paragraph(c,
                  'These examples use fictional Northstar Labs policies. Check full policies for conditions; '
                  'the app does not approve requests or determine personal balances.',
                  36, y - 2, 540, 8.5, 12, MUTED) - 16
    y = paragraph(c, 'Simple architecture', 36, y, 540, 13, 18, INK, True) - 9
    y = architecture(c, y) - 10
    y = paragraph(c,
                  'LangGraph coordinates the flow. In live mode, the hosted model plans searches, '
                  'selects excerpts and reviews the result. Python checks current sources and exact '
                  'quote support before an answer is shown. MiniLM embeddings and Chroma search stay local.',
                  36, y, 540, 8.6, 12) - 10
    c.setFillColor(HexColor('#f6f8fa'))
    c.roundRect(36, y - 41, 540, 41, 5, fill=1, stroke=0)
    paragraph(c, '<b>Optional paid fallback is off by default.</b> OpenAI or Claude can be explicitly '
                 'configured for use after transient hosted-API retries are exhausted. Rejected evidence '
                 'and terminal errors never trigger fallback.',
              46, y - 7, 520, 8.1, 11.5, MUTED)
    if y - 41 < 48:
        raise RuntimeError(f'Page 1 bottom content overflow: {y - 41:.1f}')
    page_footer(c, 1)
    c.showPage()

    # Page 2: enlarge the real answer panel, rather than shrinking a full browser window.
    y = paragraph(c, 'CP01  |  DEMO', 36, 757, 540, 10, 14, BLUE, True) - 4
    y = paragraph(c, 'See the answer in the app', 36, y, 540,
                  22, 28, '#000000', True) - 12
    y = paragraph(c, 'What home-office allowance can a new employee claim, and when?',
                  36, y, 540, 11, 16, INK, True) - 6
    y = paragraph(c, 'Real Streamlit capture in Extractive demo mode. No generation LLM was used.',
                  36, y, 540, 8.8, 12, MUTED) - 12
    im = Image.open(ROOT / '02_answer.jpg').crop((80, 236, 662, 627))
    height = 540 * im.height / im.width
    c.drawImage(ImageReader(im), 36, y - height, width=540, height=height)
    y -= height + 7
    y = paragraph(c, 'Enlarged crop of the actual answer panel. The app text and citations are unchanged.',
                  36, y, 540, 7.7, 11, MUTED) - 17
    y = paragraph(c, 'What this answer tells you', 36, y, 540,
                  13, 18, INK, True) - 8
    facts = [
        '<b>The allowance:</b> one-time USD 300 after 30 calendar days of employment.',
        '<b>The conditions:</b> line-manager approval before purchase; itemized receipts through SpendDesk.',
        '<b>The evidence:</b> [S1] is the onboarding checklist; [S2] is the equipment policy. '
        'The Evidence desk also shows each document\'s version, date and full retrieved text.',
    ]
    for fact in facts:
        y = paragraph(c, fact, 36, y, 540, 9.5, 13.5) - 7
    y = paragraph(c, 'How to read the result', 36, y - 2, 540,
                  11, 15, INK, True) - 5
    y = paragraph(c,
                  'Citations let you check where an excerpt came from. The activity trace shows the '
                  'retrieval, answer and verification steps. Exact quotations can still omit context, '
                  'so review the full policy before acting.',
                  36, y, 540, 9, 13) - 10
    y = paragraph(c,
                  'The project supports a course-hosted open-source model API; no local chat model '
                  'is required. This screenshot does not demonstrate a live API call. The prototype '
                  'is for fictional demo data and does not include production access controls.',
                  36, y, 540, 8.2, 11.5, MUTED)
    if repo_url:
        y = paragraph(c, f'<link href="{repo_url}" color="{BLUE}">Project repository</link>',
                      36, y - 8, 540, 8.2, 11.5)
    if y < 48:
        raise RuntimeError(f'Page 2 bottom content overflow: {y:.1f}')
    page_footer(c, 2)
    c.showPage()
    c.save()
    assert len(PdfReader(str(output)).pages) == 2
    print(f'PDF: {output} (2 pages)')


def pil_font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else REG), size)


def wrap(draw, text, font, max_width):
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and draw.textlength(candidate, font=font) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def make_video():
    scenes = [
        ('01_home.jpg', 'Meet CP01',
         'A local, evidence-first assistant for employee policy questions.'),
        ('02_answer.jpg', 'Ask a policy question',
         'Read the actual answer together with its supporting source citations.'),
        ('03_evidence.jpg', 'Inspect the evidence',
         'Source IDs, versions and exact policy wording stay visible for review.'),
        ('04_trace.jpg', 'Follow each decision',
         'The real LangGraph trace records retrieval, answer selection and verification.'),
        ('05_refusal.jpg', 'See the stopping rule',
         'An unrelated question returns a safe no-supported-answer response.'),
        ('06_workflow.jpg', 'Know the scope',
         'Extractive demo shown. Hosted open-source API calls were not tested. No local chat model is required.'),
    ]
    intermediate = ROOT / 'build'
    intermediate.mkdir(exist_ok=True)
    frames = []
    for i, (source, title, caption) in enumerate(scenes, 1):
        frame = Image.new('RGB', (1920, 1080), '#f4f7fb')
        d = ImageDraw.Draw(frame)
        d.text((60, 28), 'CP01  ENTERPRISE KNOWLEDGE ASSISTANT', font=pil_font(22, True), fill=BLUE)
        d.text((60, 66), title, font=pil_font(40, True), fill=INK)
        d.text((1860, 76), f'{i} / {len(scenes)}', font=pil_font(24), fill=MUTED, anchor='ra')
        im = Image.open(ROOT / source).convert('RGB')
        im.thumbnail((1800, 752), Image.Resampling.LANCZOS)
        x, y = (1920 - im.width) // 2, 143 + (752 - im.height) // 2
        frame.paste(im, (x, y))
        d.rectangle((x-1, y-1, x+im.width, y+im.height), outline='#d5deeb', width=2)
        lines = wrap(d, caption, pil_font(27), 1795)
        for j, line in enumerate(lines):
            d.text((60, 923 + 37*j), line, font=pil_font(27), fill=INK)
        d.text((60, 1031), 'Captioned walkthrough of actual app screenshots  |  Extractive demo  |  No generation LLM',
               font=pil_font(19), fill=MUTED)
        path = intermediate / f'video_frame_{i:02d}.png'
        frame.save(path)
        frames.append(path)
    concat = intermediate / 'frames.txt'
    concat.write_text(''.join(f"file '{p.name}'\nduration 10\n" for p in frames)
                      + f"file '{frames[-1].name}'\n")
    output = ROOT / 'CP01_Demo_Walkthrough.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'concat', '-safe', '0', '-i', str(concat), '-t', '60',
                    '-r', '24', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                    'format=duration,size:stream=codec_name,width,height,pix_fmt',
                                    '-of', 'json', str(output)], text=True)
    (ROOT / 'video_metadata.json').write_text(probe + '\n')
    print(f'VIDEO: {output}\n{probe}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url', help='Use only a verified repository URL')
    parser.add_argument('--pdf-only', action='store_true')
    parser.add_argument('--video-only', action='store_true')
    args = parser.parse_args()
    if not args.video_only:
        make_pdf(args.repo_url)
    if not args.pdf_only:
        make_video()
