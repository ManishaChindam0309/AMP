import { createFileRoute } from "@tanstack/react-router";
import { AmpApp } from "@/components/amp/app";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return <AmpApp />;
}
