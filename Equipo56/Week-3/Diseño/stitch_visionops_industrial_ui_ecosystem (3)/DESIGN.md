---
name: VisionOps Technical Design System
colors:
  surface: '#f8f9fa'
  surface-dim: '#d9dadb'
  surface-bright: '#f8f9fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f4f5'
  surface-container: '#edeeef'
  surface-container-high: '#e7e8e9'
  surface-container-highest: '#e1e3e4'
  on-surface: '#191c1d'
  on-surface-variant: '#414754'
  inverse-surface: '#2e3132'
  inverse-on-surface: '#f0f1f2'
  outline: '#717786'
  outline-variant: '#c1c6d7'
  surface-tint: '#005bc0'
  primary: '#0059bb'
  on-primary: '#ffffff'
  primary-container: '#0070ea'
  on-primary-container: '#fefcff'
  inverse-primary: '#adc7ff'
  secondary: '#5d5e61'
  on-secondary: '#ffffff'
  secondary-container: '#e2e2e5'
  on-secondary-container: '#636467'
  tertiary: '#755700'
  on-tertiary: '#ffffff'
  tertiary-container: '#946f00'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc7ff'
  on-primary-fixed: '#001a41'
  on-primary-fixed-variant: '#004493'
  secondary-fixed: '#e2e2e5'
  secondary-fixed-dim: '#c6c6c9'
  on-secondary-fixed: '#1a1c1e'
  on-secondary-fixed-variant: '#454749'
  tertiary-fixed: '#ffdf9e'
  tertiary-fixed-dim: '#fabd00'
  on-tertiary-fixed: '#261a00'
  on-tertiary-fixed-variant: '#5b4300'
  background: '#f8f9fa'
  on-background: '#191c1d'
  surface-variant: '#e1e3e4'
typography:
  display-lg:
    fontFamily: Hanken Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 60px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Hanken Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 30px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  gutter: 20px
  margin-mobile: 16px
  margin-desktop: 32px
---

## Brand & Style

The design system is engineered for high-stakes industrial environments where clarity, precision, and rapid data synthesis are paramount. The brand personality is authoritative, technical, and high-performance, evoking the feel of a modern digital command center.

The aesthetic utilizes a **Corporate / Modern** framework infused with **minimalist** principles to reduce cognitive load during complex monitoring tasks. It prioritizes "Information Density" over decorative flair, ensuring that Digital Twin visualizations and live video streams remain the focal point. The UI recedes into the background through the use of low-contrast UI shells, allowing critical alerts and data overlays to command immediate attention.

Target users include industrial engineers and operations managers who require a stable, reliable interface that maintains legibility under intense focus.

## Colors

The palette is bifurcated to separate global navigation from operational data. 

- **Structural Surfaces:** The sidebar and header use a "Dark Mode" logic (#1A1C1E and #2C2E33) even in the light theme to create a strong visual frame and anchor the user's focus.
- **Workspace Canvas:** Data areas utilize a pristine light environment (#F8F9FA) to ensure maximum contrast for technical charts and video overlays.
- **Functional Accents:** Electric Blue is reserved strictly for interactive elements and primary actions. Warning Yellow and Industrial Red serve as semantic signals for system health and critical alerts, respectively.

Use high-contrast ratios for all status-related text to ensure compliance with industrial safety standards.

## Typography

The typographic system balances modern sans-serif readability with technical monospaced precision.

- **Headlines:** Use **Hanken Grotesk** for a sharp, contemporary professional look. It provides high legibility at large scales for dashboard titles.
- **Body:** **Inter** is the workhorse for all data entries and descriptions, providing a neutral, systematic feel.
- **Technical Data:** **JetBrains Mono** is utilized for telemetry, timestamps, coordinates, and sensor readings. Its monospaced nature ensures that fluctuating numerical data does not cause layout shifts.

Maintain a strict vertical rhythm. All labels for sensor data or timeline markers should be rendered in uppercase monospaced type to distinguish metadata from content.

## Layout & Spacing

The layout utilizes a **Fixed Grid** system for dashboard environments to maintain the integrity of complex data visualizations. 

- **Desktop:** A 12-column grid with a max-width of 1600px. Gutters are fixed at 20px to allow for dense information displays without visual clutter.
- **Sidebars:** Persistent left-hand navigation is fixed at 240px. Right-hand "Insight" panels are collapsible, defaulting to 320px when active.
- **Spacing Rhythm:** Based on a 4px baseline grid. Use 16px (md) for standard component padding and 24px (lg) for section separation.

In Digital Twin views, the layout should maximize the viewport, utilizing "floating" UI panels with standardized 16px margins from the edge of the browser window.

## Elevation & Depth

This design system uses **Tonal Layers** and **Low-contrast outlines** to define hierarchy, avoiding heavy shadows which can muddy technical interfaces.

1.  **Level 0 (Canvas):** #F8F9FA. The base layer for all workspace activity.
2.  **Level 1 (Cards/Panels):** #FFFFFF with a 1px solid border (#E9ECEF). No shadow.
3.  **Level 2 (Overlays/Popovers):** #FFFFFF with a 1px border and a subtle, high-diffusion ambient shadow (0px 8px 24px rgba(0,0,0,0.08)).
4.  **Level 3 (Command HUD):** For video player overlays, use a semi-transparent dark blur (70% opacity #1A1C1E with 12px backdrop-blur).

Depth is communicated through color blocking rather than physical extrusion. Active states for sidebar items use a subtle 4px vertical accent bar in Electric Blue rather than a drop shadow.

## Shapes

The shape language is **Soft (0.25rem)**, emphasizing a precise, engineered feel. 

- **Standard Elements:** Buttons, inputs, and cards use a 4px corner radius.
- **Large Components:** Modal containers and video players use an 8px (rounded-lg) radius.
- **Status Indicators:** Small "Pill" shapes are used for status chips (e.g., "Active", "Offline") to provide a distinct visual contrast against the otherwise rectangular UI.

Strictly avoid circular buttons; all interactive elements should maintain a structural, rectangular footprint to align with the grid-based command center aesthetic.

## Components

- **Industrial Toggles:** Use high-contrast "Switch" patterns. When 'ON', the track should be Electric Blue; when 'OFF', a neutral mid-gray. Avoid animations longer than 150ms.
- **Tech-Overlay Video Players:** Video feeds must include a monospaced timestamp and coordinate overlay in the top-right. Controls should be minimal, utilizing the "Command HUD" glassmorphism style.
- **Technical Charts:** Lines should be 1.5px thick. Use a "Crosshair" cursor interaction that snaps to the nearest data point, displaying values in a JetBrains Mono tooltip.
- **Vertical Timelines:** Located in the right-hand panel. Use a 2px vertical track. Critical events use the Industrial Red hex for the marker and a light red tint for the background highlight of that timeline entry.
- **Input Fields:** Use "Internal Label" style for density. 1px border defaults to #DEE2E6, moving to Primary Blue on focus. Labels should transform to JetBrains Mono sm when active.
- **Data Tables:** Zebra striping is prohibited. Use 1px horizontal dividers only. Header cells must use `label-sm` typography with a #2C2E33 background for high-contrast separation.