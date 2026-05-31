import type { ShiftAiSummaryApi, ShiftSummaryApi, TimelineEventApi } from "@/lib/api";
import {
  buildReportMeta,
  buildTimelineReportHtml,
} from "@/lib/timeline-report-html";

async function renderHtmlToPdf(html: string, filename: string): Promise<void> {
  const iframe = document.createElement("iframe");
  iframe.style.position = "fixed";
  iframe.style.right = "0";
  iframe.style.bottom = "0";
  iframe.style.width = "816px";
  iframe.style.height = "0";
  iframe.style.border = "0";
  iframe.style.visibility = "hidden";
  document.body.appendChild(iframe);

  const doc = iframe.contentDocument;
  if (!doc) {
    document.body.removeChild(iframe);
    throw new Error("Could not initialize print frame");
  }

  doc.open();
  doc.write(html);
  doc.close();

  await new Promise<void>((resolve) => {
    iframe.onload = () => resolve();
    setTimeout(resolve, 800);
  });

  const target = doc.querySelector(".doc") as HTMLElement | null;
  if (!target) {
    document.body.removeChild(iframe);
    throw new Error("Report template missing .doc root");
  }

  const html2canvas = (await import("html2canvas")).default;
  const { jsPDF } = await import("jspdf");

  const canvas = await html2canvas(target, {
    scale: 2,
    useCORS: true,
    backgroundColor: "#ffffff",
    windowWidth: 816,
  });

  const imgData = canvas.toDataURL("image/png");
  const pdf = new jsPDF({ unit: "pt", format: "letter", orientation: "portrait" });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const imgWidth = pageWidth;
  const imgHeight = (canvas.height * imgWidth) / canvas.width;

  let heightLeft = imgHeight;
  let position = 0;

  pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
  heightLeft -= pageHeight;

  while (heightLeft > 0) {
    position -= pageHeight;
    pdf.addPage();
    pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;
  }

  pdf.save(filename);
  document.body.removeChild(iframe);
}

export async function exportTimelinePdf(
  events: TimelineEventApi[],
  summary: ShiftSummaryApi | null,
  aiSummary: ShiftAiSummaryApi | null,
  shiftLabel?: string,
  preparedBy?: string,
): Promise<void> {
  const label = shiftLabel ?? aiSummary?.shiftLabel ?? "SHIFT";
  const meta = buildReportMeta(events, aiSummary, label, preparedBy);
  const html = buildTimelineReportHtml(events, summary, aiSummary, meta);
  const filename = `visionops-post-shift-${new Date().toISOString().slice(0, 10)}.pdf`;
  await renderHtmlToPdf(html, filename);
}

export async function exportTimelineHtml(
  events: TimelineEventApi[],
  summary: ShiftSummaryApi | null,
  aiSummary: ShiftAiSummaryApi | null,
  shiftLabel?: string,
  preparedBy?: string,
): Promise<void> {
  const label = shiftLabel ?? aiSummary?.shiftLabel ?? "SHIFT";
  const meta = buildReportMeta(events, aiSummary, label, preparedBy);
  const html = buildTimelineReportHtml(events, summary, aiSummary, meta);
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `visionops-post-shift-${new Date().toISOString().slice(0, 10)}.html`;
  a.click();
  URL.revokeObjectURL(url);
}
