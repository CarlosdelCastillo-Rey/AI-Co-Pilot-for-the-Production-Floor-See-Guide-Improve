# VisionOps Infographic — Quick Start Guide

## 🚀 Get Started in 30 Seconds

### Option 1: Open in Browser (Easiest)

```bash
# macOS
open vision-ops-infographic/index.html

# Linux
xdg-open vision-ops-infographic/index.html

# Windows
start vision-ops-infographic/index.html
```

Or simply **drag and drop** `index.html` into your browser.

### Option 2: Serve Locally

```bash
cd vision-ops-infographic
python -m http.server 8080
# Visit http://localhost:8080
```

### Option 3: Deploy Online

```bash
# Quick deploy to Vercel (free)
npm install -g vercel
cd vision-ops-infographic
vercel

# Or deploy to Netlify
# Drag & drop folder to https://app.netlify.com
```

---

## 📋 What You Get

A **single HTML file** (`index.html`) containing:

✅ **Professional design** — VisionOps brand colors, Tailwind CSS  
✅ **Animated sections** — Scroll reveals, counter animations  
✅ **Complete pipeline** — All 5 training stages with code examples  
✅ **Real metrics** — 3,425 clips, 12 classes, 41.5% accuracy (DINOv2)  
✅ **Mathematical rigor** — Focal Loss, Mixup, Embedding formulas  
✅ **API examples** — Real curl commands and response payloads  
✅ **Mobile responsive** — Works on phone, tablet, desktop  
✅ **Self-contained** — No build process, no Node.js required  

---

## 🎯 What Each Section Covers

| Section | Content | Time |
|---------|---------|------|
| **Hero** | Mission, key stats, call-to-action | 30s |
| **Pipeline Step 01** | Data inventory (3,425 clips, 12 classes, imbalance) | 2m |
| **Pipeline Step 02** | Embedding extraction (V-JEPA, DINOv2, math) | 3m |
| **Pipeline Step 03** | Training (Focal Loss, Mixup, WeightedSampler) | 3m |
| **Pipeline Step 04** | Validation metrics (41.5% DINOv2, per-class F1) | 2m |
| **Pipeline Step 05** | Live inference (70-95ms latency breakdown) | 2m |
| **Architecture** | Backend API, Frontend pages, integrations | 2m |
| **Results** | Training metrics, inference performance table | 1m |
| **API Examples** | Real endpoints and cURL code | 2m |
| **Tech Stack** | All libraries and frameworks used | 1m |
| **Math Foundation** | Focal Loss, Weighted Sampling, Mixup formulas | 2m |
| **FAQ** | Common questions answered | 1m |

**Total read time:** ~21 minutes (skim to 5 minutes)

---

## ✏️ Customize in 5 Minutes

### Update Your Metrics

Edit `index.html` around line **380–385**:

```html
<!-- Change these three numbers -->
<div class="metric-value">3,425</div>    <!-- Your dataset size -->
<div class="metric-value">12</div>       <!-- Your classes -->
<div class="metric-value" style="color: var(--color-success);">41.5%</div>  <!-- Your accuracy -->
```

### Update Your Colors

Edit the `:root` section (lines **20–35**):

```css
:root {
    --color-primary: #0059bb;         /* Change to your brand blue */
    --color-success: #1f8a5b;         /* Change success color */
    /* ... rest stays same ... */
}
```

### Update Company Name

```bash
# One-liner to replace all instances
sed -i '' 's/Alignity IQ Edge/Your Company/g' index.html
sed -i '' 's/VisionOps/Your Product/g' index.html
```

**Done!** No build process, no npm install. Just save and refresh.

---

## 🔗 Integration with Dashboard

### Add to Next.js App (3 steps)

**Step 1:** Copy file to public folder
```bash
cp vision-ops-infographic/index.html vision-ops-app/public/infographic.html
```

**Step 2:** Create new route at `vision-ops-app/app/infographic/page.tsx`:
```tsx
export default function InfographicPage() {
  return (
    <iframe 
      src="/infographic.html"
      className="w-full h-screen"
      title="VisionOps Infographic"
    />
  );
}
```

**Step 3:** Access at `http://localhost:3000/infographic`

---

## 📊 Share & Export

### Export to PDF
```bash
# Using Chrome
google-chrome --headless --print-to-pdf=infographic.pdf \
  vision-ops-infographic/index.html
```

### Share via Link
- **GitHub Pages:** Push to repo, share raw GitHub link
- **Vercel:** `vercel deploy` → get instant URL
- **Netlify:** Drag folder to netlify.app
- **Email:** Attach HTML file (works offline)

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Animations not playing | Clear browser cache (Cmd+Shift+R) |
| Fonts wrong | Check Google Fonts loads in DevTools Network tab |
| Colors don't match | Verify `:root` CSS variables are in `<style>` |
| Mobile layout broken | Test in DevTools Device Mode (Cmd+Shift+M) |
| File too large | It's only 95 KB—perfectly fine |

---

## 📖 Full Documentation

- **README.md** — Architecture overview, all sections explained
- **CUSTOMIZATION.md** — Step-by-step advanced customization guide
- **index.html** — Inline code comments throughout

---

## 🎓 For Academic Use

Perfect for:
- ✅ Conference presentations (share HTML link)
- ✅ Journal papers (export to PDF, embed figures)
- ✅ Thesis/dissertations (complete technical infographic)
- ✅ Investor pitches (professional demo)
- ✅ Team onboarding (comprehensive technical overview)

---

## 💡 Pro Tips

1. **Print-friendly:** Browser → Print (Cmd/Ctrl+P) → Save as PDF preserves formatting
2. **Embed images:** Add `<img>` tags anywhere for custom diagrams
3. **Live data:** Add `fetch()` calls to connect backend metrics in real-time
4. **Dark mode:** Add CSS media query for `prefers-color-scheme: dark`
5. **Offline:** Works completely offline—no external API calls required

---

## 📞 Questions?

Refer to:
1. Inline HTML comments in `index.html`
2. **README.md** for full architecture
3. **CUSTOMIZATION.md** for how-to guides
4. VisionOps team slack/email for implementation help

---

## ✨ What's Next?

- [ ] Open `index.html` in browser
- [ ] Scroll through and read sections
- [ ] Update your metrics (5 min)
- [ ] Customize colors/branding (5 min)
- [ ] Share link with team
- [ ] Integrate with dashboard (optional, 15 min)
- [ ] Export to PDF for presentations

**That's it!** You now have a professional, animated academic infographic ready to share. 🚀

---

**Version:** 1.0 | **Last Updated:** June 2026 | **License:** Private (© Alignity IQ Edge, LLC)
