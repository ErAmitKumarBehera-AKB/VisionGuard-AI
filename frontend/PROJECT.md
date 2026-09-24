# VisionInspect AI

Manufacturing visual quality inspection platform with a backend-ready React client.

## Run

- `pnpm --filter @workspace/api-server run dev` runs the API server.
- `pnpm --filter @workspace/visioninspect-ai run dev` runs the frontend.
- `pnpm --filter @workspace/visioninspect-ai run typecheck` checks the frontend.
- `pnpm run build` typechecks and builds the workspace.

The frontend uses `src/services/api.ts` for authentication, inspections, feedback, and notifications. The UI renders empty states until the corresponding API endpoints provide data; no local records, timestamps, or simulated inference are bundled.

## Stack

- pnpm workspaces, Node.js, TypeScript
- Express API server
- PostgreSQL and Drizzle ORM
- React, Vite, Tailwind CSS, Wouter, Recharts, and Lucide React

## Key files

- `artifacts/visioninspect-ai/src/App.tsx` contains routes and screens.
- `artifacts/visioninspect-ai/src/data/types.ts` contains shared client types and empty initial collections.
- `artifacts/visioninspect-ai/src/services/api.ts` contains backend request boundaries.
- `artifacts/visioninspect-ai/src/index.css` contains the visual system.

## Integration endpoints

The client expects authenticated endpoints under `/api` for session management, inspections, feedback, and notifications. Add the matching Express routes and persistence layer in `artifacts/api-server` before enabling production workflows.
