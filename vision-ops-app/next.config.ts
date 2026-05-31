import type { NextConfig } from "next";

const backendUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const alertingUrl = process.env.NEXT_PUBLIC_ALERTING_URL ?? "http://localhost:8001";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: "/vision-api/:path*",
        destination: `${backendUrl}/:path*`,
      },
      {
        source: "/alerting-api/:path*",
        destination: `${alertingUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
