import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Nav from "@/components/Nav";
import ServiceWorker from "@/components/ServiceWorker";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const description = "Offline diabetic retinopathy screening on a low-cost phone, with on-device AI that shows its reasoning.";

// Link previews need an absolute image URL. Vercel sets the production domain
// at build time, so renaming it needs no code change.
const site = process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(site),
  title: "Netra · DR screening",
  description,
  applicationName: "Netra",
  appleWebApp: { capable: true, title: "Netra", statusBarStyle: "default" },
  openGraph: {
    type: "website",
    siteName: "Netra",
    title: "Netra: offline diabetic retinopathy screening",
    description,
    images: [{ url: "/og.png", width: 1200, height: 627, alt: "Netra on a phone: refer to an eye specialist, with a heatmap over the retina" }],
  },
  twitter: { card: "summary_large_image", images: ["/og.png"] },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0c0f17" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Nav />
        <main className="mx-auto w-full max-w-3xl px-4 pb-28 pt-4 sm:pb-12">{children}</main>
        <ServiceWorker />
      </body>
    </html>
  );
}
