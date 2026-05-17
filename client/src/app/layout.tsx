import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Ask my cv - Simone Bitti",
  description: "Ask my cv is a web application that allows you to interact with my CV using natural language. You can ask questions about my skills, experience, and education, and get answers in real-time.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable} h-full`}>
      <head>
          <link rel="icon" href="favicon.ico" type="icon"/>
          <meta charSet="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content" />
      </head>
      <body className="min-h-screen flex flex-col bg-cream text-ink antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
