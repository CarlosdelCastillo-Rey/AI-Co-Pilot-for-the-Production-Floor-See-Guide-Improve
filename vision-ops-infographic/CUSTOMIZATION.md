# VisionOps Infographic — Customization & Integration Guide

## Overview

This guide walks you through customizing the academic infographic for your specific metrics, branding, and integration with the VisionOps dashboard.

---

## 1. Quick Customizations (5 minutes)

### Update Core Metrics

**File:** `index.html`

Find and replace these sections:

```html
<!-- SECTION: Hero Statistics -->
<!-- Line ~380: Update data counts -->
<div class="metric-value">3,425</div>  <!-- Training clips -->
<p class="text-sm text-gray-600 mt-2 font-semibold">Industrial Video Clips</p>

<div class="metric-value">12</div>  <!-- Classes -->
<p class="text-sm text-gray-600 mt-2 font-semibold">Action Classes</p>

<div class="metric-value" style="color: var(--color-success);">41.5%</div>  <!-- Accuracy -->
```

### Update Backbone Performance

Find the Results table (around line 680):

```html
<tr>
    <td>DINOv2 (ViT-L/14)</td>
    <td class="value-good">41.5%</td>      <!-- Update accuracy -->
    <td class="value-good">0.425</td>      <!-- Update macro F1 -->
    <td class="value-good">0.418</td>      <!-- Update weighted F1 -->
</tr>
```

### Update Per-Class Scores

Find the performance cards (around line 750):

```html
<div class="bg-green-50 p-3 rounded-lg border border-green-200">
    <p class="text-xs font-semibold text-green-900">Turn sheets</p>
    <p class="text-lg font-bold text-green-700">0.57</p>  <!-- Update F1 -->
</div>
```

**Change card background** to reflect performance:
- `.bg-green-50` — High F1 (>0.45)
- `.bg-blue-50` — Medium F1 (0.30–0.45)
- `.bg-yellow-50` — Low F1 (<0.30)

---

## 2. Brand Customization (15 minutes)

### Update Color Scheme

Edit the `:root` section (lines 20–35):

```css
:root {
    /* Change primary color */
    --color-primary: #0059bb;              /* ← Your brand blue */
    --color-primary-container: #0070ea;    /* ← Lighter shade */
    
    /* Change accent colors */
    --color-success: #1f8a5b;              /* ← Your success green */
    --color-warning: #b7791f;              /* ← Your warning orange */
    --color-error: #ba1a1a;                /* ← Your error red */
}
```

**Example:** If your brand is purple:

```css
--color-primary: #6b21a8;        /* Deep purple */
--color-primary-container: #7c3aed;  /* Vivid purple */
--color-success: #059669;        /* Keep green for "good" metrics */
```

### Update Company Name

Replace all instances of:
- `Alignity IQ Edge` → Your company
- `VisionOps` → Your product name
- `research@alignityiq.com` → Your email

**Quick find/replace:**
```bash
sed -i '' 's/VisionOps/YourProduct/g' index.html
sed -i '' 's/Alignity IQ Edge/YourCompany/g' index.html
```

### Update Logo / Favicon

Add to the `<head>` section:

```html
<link rel="icon" href="/logo.ico" type="image/x-icon">
<link rel="apple-touch-icon" href="/logo-192.png">
```

---

## 3. Data Updates (20 minutes)

### Dataset Statistics

Update the data pipeline section (around line 430):

```html
<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
    <div>
        <div class="font-bold text-lg">3,425</div>   <!-- Total clips -->
        <div class="text-gray-500 text-xs">Total clips</div>
    </div>
    <div>
        <div class="font-bold text-lg">2,924</div>   <!-- Training -->
        <div class="text-gray-500 text-xs">Training</div>
    </div>
    <div>
        <div class="font-bold text-lg">501</div>     <!-- Validation -->
        <div class="text-gray-500 text-xs">Validation</div>
    </div>
    <div>
        <div class="font-bold text-lg">12 subj.</div> <!-- Subjects -->
        <div class="text-gray-500 text-xs">Operators</div>
    </div>
</div>
```

### Model Architecture Details

Update the backbone comparison section (around line 485):

```html
<div class="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
    <p class="font-label text-xs text-blue-600 mb-2">YOUR BACKBONE NAME</p>
    <p class="text-sm font-semibold mb-1">Model ID</p>
    <ul class="text-xs text-gray-700 space-y-1">
        <li>• Feature 1</li>
        <li>• Feature 2</li>
    </ul>
</div>
```

### Training Configuration

Update the code block (around line 545):

```html
<pre><span class="keyword">trainer</span> <span class="operator">=</span> <span class="function">HarTrainer</span>(<br/>
    head_arch<span class="operator">=</span><span class="string">"mlp"</span>,<br/>
    epochs<span class="operator">=</span><span class="number">100</span>,<br/>
    batch_size<span class="operator">=</span><span class="number">64</span><br/>
<span class="operator">)</span></pre>
```

### Inference Latency Budget

Update the latency breakdown (around line 810):

```html
<div style="--delay: 0.4s;">
    <div class="flex items-start gap-6">
        ...
        <div class="bg-gray-50 p-4 rounded-lg font-label text-xs">
            <p class="font-label text-xs text-gray-500 mb-3">LIVE INFERENCE PIPELINE</p>
            <div class="space-y-2 text-sm font-mono text-gray-700">
                <div>1. Read frame from MP4 mock or RTSP stream</div>
                <div>2. YOLO person detection → bounding boxes</div>
                <div>3. ByteTrack: assign persistent track IDs</div>
                <!-- Update these steps based on your pipeline -->
            </div>
        </div>
```

---

## 4. Advanced Customization

### Add Custom Sections

To insert a new section (e.g., "Our Innovations"), place it between dividers:

```html
<div class="section-divider"></div>

<section class="py-20 px-4 bg-white">
    <div class="max-w-6xl mx-auto">
        <h2 class="font-headline text-4xl mb-12 text-center">Your Section Title</h2>
        
        <div class="card fade-in-on-scroll">
            <h3 class="font-headline text-xl mb-4">Subsection</h3>
            <p class="text-gray-600 text-sm mb-4">Your content here</p>
        </div>
    </div>
</section>

<div class="section-divider"></div>
```

### Modify Pipeline Steps

To add a Step 06 or rename existing steps:

```html
<!-- Original Step 05 -->
<div class="card fade-in-on-scroll" style="--delay: 0.4s;">
    <div class="flex items-start gap-6">
        <div class="flex-shrink-0 w-12 h-12 rounded-full bg-orange-100 flex items-center justify-center font-bold text-orange-600">05</div>
        <!-- ... -->
    </div>
</div>

<!-- New Step 06 -->
<div class="card fade-in-on-scroll" style="--delay: 0.5s;">
    <div class="flex items-start gap-6">
        <div class="flex-shrink-0 w-12 h-12 rounded-full bg-red-100 flex items-center justify-center font-bold text-red-600">06</div>  <!-- New number, new color -->
        <div class="flex-grow">
            <h3 class="font-headline text-xl mb-2">Your Step Title</h3>
            <p class="text-gray-600 text-sm">Your description</p>
        </div>
    </div>
</div>
```

### Embed Live Data

To connect to backend for real-time metrics, add a fetch call in the `<script>` section:

```javascript
// At the end of the <script> section before closing tag
async function updateMetrics() {
    try {
        const response = await fetch('http://localhost:8000/api/analytics/summary');
        const data = await response.json();
        
        // Update the metric displays
        document.querySelector('.metric-value').textContent = data.accuracy.toFixed(1) + '%';
    } catch (error) {
        console.error('Failed to fetch metrics:', error);
    }
}

// Call on page load
window.addEventListener('load', updateMetrics);

// Refresh every 30 seconds
setInterval(updateMetrics, 30000);
```

---

## 5. Integration with Vision-Ops Dashboard

### Option A: Embed as IFrame

In `vision-ops-app/app/(dashboard)/layout.tsx` or any page, add:

```tsx
import { useEffect } from 'react';

export default function InfographicModal() {
  return (
    <div className="fixed inset-0 bg-black/50 z-50 overflow-auto">
      <iframe 
        src="/infographic.html"
        className="w-full h-full border-0"
        title="VisionOps Technical Infographic"
        sandbox="allow-same-origin allow-scripts"
      />
    </div>
  );
}
```

### Option B: Create Dedicated Route

Create `vision-ops-app/app/infographic/page.tsx`:

```tsx
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'VisionOps Technical Infographic',
  description: 'Complete HAR pipeline visualization and metrics',
};

export default function InfographicPage() {
  return (
    <div className="w-full">
      <iframe 
        src="/infographic.html"
        className="w-full h-screen border-0"
        title="VisionOps Technical Infographic"
      />
    </div>
  );
}
```

Then copy `index.html` to `vision-ops-app/public/infographic.html`:

```bash
cp vision-ops-infographic/index.html vision-ops-app/public/infographic.html
```

Access at: `http://localhost:3000/infographic`

### Option C: Component Extraction

Convert HTML sections to React components in `vision-ops-app/components/infographic/`:

```tsx
// components/infographic/PipelineSteps.tsx
import { motion } from 'framer-motion';

export function PipelineSteps() {
  return (
    <section className="py-20 px-4 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold mb-12 text-center">The Complete HAR Pipeline</h2>
        
        {/* Steps */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          {/* Step content */}
        </motion.div>
      </div>
    </section>
  );
}
```

---

## 6. Keeping Data in Sync

### Automated Metric Updates

Create a script to auto-update metrics from backend:

```python
# scripts/sync_infographic_metrics.py
import json
import subprocess
from datetime import datetime

# Query backend for latest metrics
response = requests.get('http://localhost:8000/api/analytics/summary')
data = response.json()

# Read infographic
with open('vision-ops-infographic/index.html', 'r') as f:
    html = f.read()

# Replace metrics
html = html.replace(
    '<div class="metric-value" style="color: var(--color-success);">41.5%</div>',
    f'<div class="metric-value" style="color: var(--color-success);">{data["accuracy"]:.1f}%</div>'
)

# Write back
with open('vision-ops-infographic/index.html', 'w') as f:
    f.write(html)

print(f"Updated metrics at {datetime.now()}")
```

Run on schedule:

```bash
# In crontab: Update every morning at 2am
0 2 * * * cd /path/to/project && python scripts/sync_infographic_metrics.py
```

---

## 7. Export & Distribution

### Export to PDF (Maintain Interactivity)

Use Chrome headless:

```bash
# Full page PDF
google-chrome --headless --disable-gpu --print-to-pdf=infographic.pdf vision-ops-infographic/index.html

# Specific dimensions (for printing)
google-chrome --headless --disable-gpu --print-to-pdf=infographic-print.pdf \
  --print-to-pdf-margin-top=0.5 \
  --print-to-pdf-margin-bottom=0.5 \
  vision-ops-infographic/index.html
```

### Create Printable Version

Add CSS for printing. Insert before closing `</style>`:

```css
@media print {
    body {
        background: white;
        padding: 0;
    }

    .fade-in-on-scroll {
        opacity: 1 !important;
        animation: none !important;
    }

    section {
        page-break-inside: avoid;
    }

    .section-divider {
        display: none;
    }
}
```

Then print via browser: `Cmd/Ctrl + P` → Save as PDF

### Deploy to Web

**Option 1: Vercel (Recommended)**

```bash
# Create vercel.json
cat > vercel.json << 'EOF'
{
  "buildCommand": "echo 'Static site'",
  "outputDirectory": "vision-ops-infographic"
}
EOF

vercel deploy
```

**Option 2: GitHub Pages**

```bash
# Push to gh-pages branch
git subtree push --prefix vision-ops-infographic origin gh-pages
# Access at: https://yourusername.github.io/AI-Co-Pilot.../vision-ops-infographic/
```

**Option 3: AWS S3 + CloudFront**

```bash
# Upload and set as static website
aws s3 sync vision-ops-infographic/ s3://your-bucket/infographic/ --acl public-read
```

---

## 8. Troubleshooting

### Animations Not Playing

**Problem:** Scroll animations or counters don't trigger

**Solution:** Check GSAP library loaded:
```javascript
// In DevTools Console:
console.log(window.gsap);  // Should not be undefined
```

If undefined, reload with DevTools open to check network tab for GSAP CDN.

### Colors Don't Match Design

**Problem:** CSS variables not applying

**Solution:** Ensure `:root` is set before any components:
```css
:root {
    /* Must be FIRST in <style> block */
    --color-primary: #0059bb;
    /* ... rest of variables ... */
}
```

### Mobile Layout Broken

**Problem:** Content misaligned on phones

**Solution:** Test in DevTools mobile mode, ensure Tailwind breakpoints are used:
```html
<!-- Mobile-first -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
```

### Font Not Loading

**Problem:** Text appears in default font

**Solution:** Check Google Fonts link:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
```

If blocked, use system fonts:
```css
.font-headline {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
```

---

## Support

For questions or issues:

1. **Check inline comments** in `index.html` — many sections are documented
2. **Refer to README.md** for architecture overview
3. **Review vision-ops-app styles** — see `globals.css` for design tokens
4. **Ask VisionOps team** — reach out for customization help

---

**Happy customizing!** 🚀
