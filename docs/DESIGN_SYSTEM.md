# Design System

## 1. Design Direction
- **Style:** Technical, Minimalist, Engineering-Grade (Monospace-accented, clean dark/light mode).
- **Mood:** Deterministic, reliable, high-performance, calm, focused.
- **Design Principles:**
  - *Clarity Over Decoration:* Emphasize content hierarchy, code formatting, and confidence metrics over superfluous visuals.
  - *Radical Transparency:* Never hide the AI's internal process; display retrieval candidates, scores, and reflection steps clearly.
  - *Bilingual Fluency:* Seamless Bi-directional (RTL for Persian, LTR for English/Code) alignment without layout breakage.

---

## 2. Brand
- **Brand Name:** OmniDoc RAG
- **Logo:** Monospace geometric glyph: `[Ω / RAG]` rendered in Emerald/Slate.
- **Brand Voice:** Objective, concise, technical, evidence-grounded.

---

## 3. Color Palette
- **Primary (Accent Brand):** `#10B981` (Emerald 500 - System active, high relevance)
- **Primary Hover:** `#059669` (Emerald 600)
- **Secondary:** `#6366F1` (Indigo 500 - Cross-Encoder / Reranking accents)
- **Background (Dark):** `#090D16` (Deep Obsidian Slate)
- **Background (Light):** `#F8FAFC` (Slate 50)
- **Surface / Card (Dark):** `#131B2E` (Navy Slate Container)
- **Surface / Card (Light):** `#FFFFFF` (Pure White Container)
- **Text (Dark Mode):** `#F1F5F9` (Slate 100)
- **Text (Light Mode):** `#0F172A` (Slate 900)
- **Muted Text:** `#94A3B8` (Slate 400)
- **Border:** `#1E293B` (Dark) / `#E2E8F0` (Light)
- **Success:** `#22C55E` (Score > 0.70 / Relevant)
- **Warning:** `#F59E0B` (Score 0.35–0.70 / Ambiguous Context)
- **Error:** `#EF4444` (Score < 0.35 / Inadequate Context / Network Failure)

---

## 4. Typography
- **Primary Persian Font:** `Vazirmatn`, sans-serif (Weights: 400 Regular, 500 Medium, 700 Bold)
- **Primary English Font:** `Inter`, -apple-system, sans-serif
- **Code & Monospace Font:** `JetBrains Mono`, `Fira Code`, monospace
- **Scale:**
  - **H1:** 28px / Weight: 700 / Line Height: 1.3
  - **H2:** 22px / Weight: 600 / Line Height: 1.35
  - **H3:** 18px / Weight: 600 / Line Height: 1.4
  - **Body:** 15px / Weight: 400 / Line Height: 1.6
  - **Small / Meta:** 12px / Weight: 500 / Line Height: 1.5
  - **Code Blocks:** 13px / Weight: 400 / Line Height: 1.5

---

## 5. Spacing Scale (4px Base Unit)
- **xs:** 4px
- **sm:** 8px
- **md:** 12px
- **base:** 16px
- **lg:** 24px
- **xl:** 32px
- **2xl:** 48px
- **3xl:** 64px

---

## 6. Border Radius
- **Small (Tags, Badges, Micro-chips):** 4px
- **Medium (Buttons, Input Fields):** 8px
- **Large (Containers, Thought Accordions, Modals):** 12px
- **Pill (Status Badges, Filter Pills):** 9999px

---

## 7. Shadows
- **Card Subtle:** `0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px -1px rgba(0, 0, 0, 0.3)`
- **Elevation / Popover:** `0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.4)`
- **Glow Highlight (Accent):** `0 0 15px rgba(16, 185, 129, 0.25)`

---

## 8. Buttons
- **Primary Action:** Solid Emerald background (`#10B981`), white text, 8px radius, 12px vertical padding. Focus ring with 2px offset.
- **Secondary Action:** Slate border (`#334155`), transparent fill, hover subtle slate fill.
- **Danger Action:** Solid Rose/Red (`#EF4444`), white text.
- **Disabled State:** Opacity 0.45, cursor `not-allowed`, no hover transforms.
- **Loading State:** Spinning dual-ring SVG indicator with accessible `aria-busy="true"`.

---

## 9. Inputs & Search Box
- **Container:** Rounded 10px, border 1px solid `#334155`.
- **Focus:** Border `#10B981`, ring 2px `#10B98133`.
- **Placeholder:** `#64748B`, font-style normal, RTL-adaptive direction.
- **Actions inside input:** Attachment button, Clear button, Submit icon button.

---

## 10. Cards & Thought Accordions (Chainlit Steps)
- **Thought Step Container:** Collapsible accordion with border `#1E293B` and surface `#131B2E`.
- **Step Header:** Icon (Search / Filter / Sparkles) + Label ("Dense Search", "Reranker Scoring", "CRAG Reflection") + Status Badge (Duration in ms).
- **Step Body:** Formatted JSON or ranked list of chunks with source file and similarity score pill.

---

## 11. States
- **Loading:** Skeleton loader with pulse animation (`#1E293B` to `#334155`).
- **Empty State:** Centered icon with helpful query prompts (e.g., "خطای اتصال به دیتابیس پستگرس چگونه رفع می‌شود؟").
- **Error State:** Alert banner with clear human-readable error description and retry button.
- **Stream State:** Blinking vertical cursor `|` following the last streamed token.

---

## 12. Icons
- **Library:** Lucide Icons (`lucide-react` / Chainlit native icon set).
- **Default Size:** 16px (in-line), 20px (action buttons), 24px (headers).

---

## 13. Accessibility & Bi-directional Support (RTL/LTR)
- Automatic text direction detection: If first non-punctuation character is Persian/Arabic, container renders `dir="rtl"`; English technical code blocks inside render strictly `dir="ltr"`.
- Contrast ratio: Minimum 4.5:1 for body text against dark/light surfaces (WCAG AA compliant).
- Full keyboard tab navigation across input fields, source citations, and collapsible steps.
