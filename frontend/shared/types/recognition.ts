/** Nuxt same-origin proxy -> real wine_pipeline async API. */
export interface WineCandidate {
  wine_id: string;
  slug: string;
  score: number;
  distance?: number;
  best_image_uri: string;
  name?: string;
  winery?: string;
  region?: string;
  grape_variety?: string;
  color?: string;
  category?: string;
  description?: string;
  wine_url?: string;
  web_photo_uri?: string | null;
  vintage?: number | null;
  attributes?: { label: string; value: string }[];
}
export interface VintageCheck {
  detected_year: number | null;
  state: "not_read" | "ambiguous" | "matched" | "unverified" | "mismatch";
}
/** Advisory ranking diagnostics; the score is not a calibrated probability. */
export interface MatchDecision {
  accepted: boolean;
  calibrated: boolean;
  score: number;
  margin: number | null;
  reason: string;
  policy_version: string;
}
export interface SearchResponse {
  request_id: string;
  status: "ok" | "no_results";
  model_name: string;
  query_embedding_dimension: number;
  candidates: WineCandidate[];
  slug?: string;
  message?: string | null;
  warnings?: string[];
  fusion_mode?: string;
  fusion_weights?: { visual: number; ocr: number };
  vintage_check?: VintageCheck;
  decision?: MatchDecision;
}
export type JobState = "queued" | "processing" | "done" | "failed";
export type JobStage = "prepare" | "preprocess" | "preprocess_done" | "search" | "rank" | "dino_started" | "dino_done"
  | "superpoint_started" | "superpoint_done" | "superpoint_failed"
  | "ocr_started" | "ocr_done" | "ocr_failed" | "ocr_rescue_started" | "ocr_rescue_done" | "ocr_rescue_failed"
  | "fusion" | "color" | "vintage" | "done" | "failed";
export interface JobAccepted { job_id: string; state: JobState; poll_after_ms?: number }
export interface JobError { code: string; message: string }
export interface JobStatus {
  job_id: string;
  state: JobState;
  stage: JobStage | null;
  progress: number;
  poll_after_ms?: number;
  result: SearchResponse | null;
  error: JobError | null;
  history?: { stage: JobStage; progress: number; elapsed_ms: number }[];
}
