# VisionOps Technical Infographic

## Overview

A professional, animated academic infographic presenting the complete VisionOps HAR (Human Action Recognition) pipeline—from raw industrial video data to production AI inference. Designed for presentations, publications, and stakeholder communication.

**Live Demo:** Open `index.html` in any modern web browser

### Features

✅ **Fully Self-Contained** — Single HTML file, no build process required  
✅ **Animated Sections** — Scroll-triggered reveals, counter animations  
✅ **Scientific Content** — Complete mathematical derivations, real metrics  
✅ **Production-Ready** — Integrated with VisionOps design tokens (Tailwind CSS)  
✅ **Responsive Design** — Mobile, tablet, desktop optimized  
✅ **Real Data** — Actual pipeline metrics (41.5% DINOv2 accuracy, 3,425 clips, etc.)

---

## What's Inside

### Section 1: Hero & Executive Summary
- Project tagline: "Visual Intelligence for Industrial Floors"
- Key statistics: 3,425 clips, 12 action classes, 41.5% accuracy
- Mission statement and value proposition

### Section 2: Complete HAR Pipeline (5 Steps)

**Step 01: Data Inventory & Strategy**
- InHARD dataset analysis (3,425 clips, 12 classes)
- Class imbalance identification (16.4× ratio)
- Subject-aware train/test split strategy

**Step 02: Frozen Embedding Extraction**
- Two-backbone comparison (V-JEPA 2 vs. DINOv2)
- YOLO top-down crop pipeline
- Temporal pooling mathematical formulation
- 1024-dimensional embedding output

**Step 03: Multi-Improvement Training**
- Focal Loss (γ=2.0) for imbalance
- Weighted Random Sampler (inverse class frequency)
- Mixup augmentation (2,924 → 8,766 synthetic samples)
- CosineAnnealingLR schedule
- Code example included

**Step 04: Validation & Analysis**
- Per-backbone metrics (DINOv2 vs. V-JEPA)
- Per-class F1 scores (Turn sheets: 0.57, Picking left: 0.20)
- Class distribution correlation with performance

**Step 05: Real-Time Inference**
- Live pipeline: YOLO → ByteTrack → DINOv2 → MLP
- Per-frame latency budget (70-95ms)
- SQLite activity logging
- Alert rule dispatch flow

### Section 3: Production Architecture

**Backend (`:8000`)**
- Unified FastAPI service
- In-process HAR inference
- SQLite activity stores
- Alert classification (Strands + Ollama)
- API endpoints with examples

**Frontend (`:3000`)**
- Next.js 16 + React 19 dashboard
- /analytics, /live, /timeline, /har-hitl pages
- VisionOps AI Advisor (floating button)
- Real-time KPI tracking

### Section 4: Quantified Results

**Training Metrics**
- Dataset size progression
- Training time on GPU
- Epoch-based training curves
- Early stopping configuration

**Inference Performance**
- YOLO: 8-12ms
- ByteTrack: 2-3ms
- DINOv2 embedding: 40-60ms
- MLP: <2ms
- **Total E2E: 70-95ms** ✓ sub-100ms latency

**Per-Action Insights**
- Rare actions (Picking left: 31 samples) → low F1 (0.20)
- Common actions (Turn sheets: 512 samples) → high F1 (0.57)
- Visual correlation between sample count and performance

### Section 5: Technology Stack

**Vision Models**
- DINOv2 (ViT-L/14) — Primary
- V-JEPA 2 — Fallback
- YOLOv8 — Person detection
- ByteTrack — Temporal tracking
- SFace — Face recognition

**Backend & Data**
- FastAPI, SQLite, Ollama, JWT Auth
- MailerSend, Telegram integration
- PyTorch + torchvision training

**Frontend**
- Next.js 16, React 19, Tailwind CSS 4
- Framer Motion, Strands AI

**Research**
- Jupyter Notebooks (00–08)
- InHARD dataset (5,303 clips)
- UMAP/t-SNE analysis

### Section 6: Mathematical Foundation

**Embedding Extraction Formula**
```
e_clip = TemporalPool(Backbone([crop_1, ..., crop_16]))
where crop_i ∈ ℝ^(3×224×224) from YOLO
output: e_clip ∈ ℝ^1024
```

**Focal Loss**
```
L_focal = -α_t · (1-p_t)^γ · log(p_t)
γ=2.0 focuses on hard examples
```

**Weighted Sampling**
```
w_c = 1 / √(n_c)
Balances mini-batches by inverse frequency
```

**Mixup Augmentation**
```
λ ~ Beta(α, α), α=0.2
x̃ = λ·x_i + (1-λ)·x_j
2,924 originals × 3 aug = 8,766 training samples
```

### Section 7: API Examples

Real-world integration code:
- Ingest live HAR activity (POST /api/har/activity)
- Query analytics (GET /api/har/analytics/daily)
- Timeline workflow (PATCH /api/timeline/*/acknowledge, /resolve)

### Section 8: FAQ & Troubleshooting

Common questions:
- Why DINOv2 over V-JEPA?
- How does Focal Loss help?
- What's the inference latency?
- How does HITL improve models?

---

## Quick Start

### Option 1: Direct File Access

```bash
# Open in browser
open vision-ops-infographic/index.html

# Or serve via Python
cd vision-ops-infographic
python -m http.server 8080
# Visit http://localhost:8080
```

### Option 2: Integration with Vision-Ops App

Copy to Next.js public directory for serving:

```bash
cp vision-ops-infographic/index.html vision-ops-app/public/infographic.html
```

Then access at: `http://localhost:3000/infographic.html`

### Option 3: Embed in Dashboard

Create a new route in `vision-ops-app/app/infographic/page.tsx`:

```tsx
export default function InfographicPage() {
  return (
    <iframe 
      src="/infographic.html" 
      className="w-full h-screen border-0"
      title="VisionOps Technical Infographic"
    />
  );
}
```

---

## Design System Integration

The infographic uses VisionOps design tokens from `vision-ops-app/app/globals.css`:

| Token | Value | Usage |
|-------|-------|-------|
| `--color-primary` | `#0059bb` | Main buttons, highlights |
| `--color-success` | `#1f8a5b` | Positive metrics (F1, accuracy) |
| `--color-warning` | `#b7791f` | Caution metrics (low F1) |
| `--color-error` | `#ba1a1a` | Critical alerts |
| `--font-headline` | Hanken Grotesk | Section titles |
| `--font-label` | JetBrains Mono | Code, metrics |
| `--font-body` | Inter | Body text |

All colors are CSS variables, making it easy to rebrand by editing the `:root` section.

---

## Customization Guide

### 1. Update Company/Product Name

**Find:**
```html
<title>VisionOps: AI Visual Intelligence...</title>
```

**Replace with your branding:**
```html
<title>YourProduct: AI Visual Intelligence...</title>
```

### 2. Change Metrics & Data

Update these sections with your actual numbers:

```html
<!-- Hero Stats -->
<div class="metric-value">3,425</div>  <!-- Change to your clip count -->
<div class="metric-value">12</div>      <!-- Change to your classes -->
<div class="metric-value" style="color: var(--color-success);">41.5%</div>  <!-- Your accuracy -->
```

### 3. Modify Color Scheme

Edit the `:root` styles at the top:

```css
:root {
    --color-primary: #0059bb;           /* Change this */
    --color-success: #1f8a5b;           /* And this */
    --color-warning: #b7791f;           /* And this */
    /* ... etc ... */
}
```

### 4. Update Results Table

Replace per-class F1 scores:

```html
<div class="bg-green-50 p-3 rounded-lg border border-green-200">
    <p class="text-xs font-semibold text-green-900">Your Action Name</p>
    <p class="text-lg font-bold text-green-700">0.XX</p>  <!-- Update score -->
</div>
```

### 5. Add Custom Sections

Insert new sections between dividers:

```html
<div class="section-divider"></div>

<section class="py-20 px-4 bg-white">
    <div class="max-w-6xl mx-auto">
        <h2 class="font-headline text-4xl mb-12 text-center">Your Section</h2>
        <!-- Content here -->
    </div>
</section>

<div class="section-divider"></div>
```

### 6. Update API Endpoints

Replace example endpoints with your actual URLs:

```html
<span class="api-method">GET</span> /api/your-endpoint<br/>
<span class="text-gray-600">Your description</span>
```

---

## Performance Notes

### File Size
- **Total:** ~95 KB (HTML only, no external dependencies except CDN)
- **CDN Dependencies:** Tailwind CSS, GSAP, Google Fonts (~200 KB, cached)
- **Total Load:** ~295 KB

### Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

### Load Time
- **First Paint:** ~500ms
- **Interactive:** ~1.2s
- **Fully Loaded:** ~2-3s (with animations)

### Optimization Tips

1. **Minify HTML** (optional):
```bash
npm install -g html-minifier
html-minifier --collapse-whitespace --remove-comments index.html > index.min.html
```

2. **Self-Host Fonts** (if offline required):
Replace Google Fonts link with local fonts in `/fonts/` directory

3. **Static Hosting** (recommended):
Deploy to Vercel, Netlify, or S3 + CloudFront

---

## Embedding in Presentations

### PowerPoint / Google Slides

1. **Screen Capture:** Use browser DevTools to export sections as PNG
   ```bash
   # In DevTools Console, capture specific card:
   html2canvas(document.querySelector('.card')).then(canvas => {
       canvas.toBlob(blob => {
           // Use blob as image
       });
   });
   ```

2. **Full Page PDF:** Print to PDF via browser
   ```
   Cmd/Ctrl + P → Save as PDF
   ```

3. **Interactive Web Link:** Share `index.html` URL directly

### Academic Paper / LaTeX

Extract individual sections as figures:

```latex
\includegraphics[width=\textwidth]{visionops-pipeline.png}
\caption{VisionOps HAR pipeline: Data → Embeddings → Training → Inference}
```

---

## Integration with Vision-Ops Backend

The infographic documents the actual production system. To keep it in sync:

### Accuracy Changes
When retraining models, update this section:

```html
<div class="metric-value" style="color: var(--color-success);">41.5%</div>
```

### New Features
Add steps to the pipeline section when adding capabilities:

```html
<!-- Step 06: New Feature -->
<div class="card fade-in-on-scroll" style="--delay: 0.5s;">
    <div class="flex items-start gap-6">
        <div class="flex-shrink-0 w-12 h-12 rounded-full bg-blue-100...">06</div>
        <!-- Content -->
    </div>
</div>
```

### API Changes
Update the Integration & API Examples section to match current endpoints

---

## Troubleshooting

### Animations not playing
- Check browser DevTools → disable extensions (ad blockers can break GSAP)
- Try Chrome instead of Safari (better GSAP support)

### Fonts not loading
- Check network tab → Google Fonts should load
- If blocked by firewall, edit CSS to use system fonts

### Layout broken on mobile
- Viewport meta tag is correct, but test in DevTools mobile mode
- Tailwind responsive classes should handle breakpoints

### Colors not matching design
- Ensure `globals.css` tokens are synced
- Check browser dark mode → disable in DevTools

---

## Future Enhancements

- [ ] Add interactive model comparison widget (V-JEPA vs. DINOv2 real-time)
- [ ] Embed live camera feed widget from `/api/cameras/stream`
- [ ] Add 3D visualization of embedding space (UMAP)
- [ ] Real-time metrics dashboard connected to backend
- [ ] A/B testing results visualization
- [ ] Export to PDF with full interactive links
- [ ] Dark mode toggle

---

## License & Attribution

**Private — © Alignity IQ Edge, LLC**

This infographic documents proprietary VisionOps technology. Sharing is permitted only under NDA. For inquiries, contact: `research@alignityiq.com`

---

## Quick Reference

| What | Where |
|------|-------|
| View live | `vision-ops-infographic/index.html` → browser |
| Serve locally | `python -m http.server 8080` in directory |
| Customize | Edit CSS variables and HTML sections |
| Embed | Add to Next.js `/public` or create new route |
| Update data | Search & replace metric values, tables |
| Export to PDF | Browser → Print → Save as PDF |

---

**Questions?** Refer to the inline code comments or reach out to the VisionOps team.
