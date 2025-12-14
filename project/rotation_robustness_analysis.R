# rotation_robustness_analysis.R
# New script to compute rotation robustness metrics (TopK overlap, cosine similarity, score stability)
# and prepare qualitative DenseFT pairings without touching existing Python pipelines.
# Requirements: R packages jsonlite, reticulate, dplyr, purrr, tidyr, stringr, glue, ggplot2.
# Optional (for image grids): magick, patchwork/cowplot.
# Data layout: raw variants stay on D:/dinov3_exports, while subsampled/top-k variants
# live on H:/dinov3_exports with underscored directory names (e.g., _vits16+/_100/_045/_PatchGrid).

suppressPackageStartupMessages({
  library(jsonlite)
  library(reticulate)
  library(dplyr)
  library(purrr)
  library(tidyr)
  library(stringr)
  library(glue)
  library(ggplot2)
})

# --- Config ------------------------------------------------------------------
config <- list(
  dataset    = "shinsung_data",
  roots = list(
    raw = list(
      orig = "D:/dinov3_exports/dinov3_embeds",
      query = "D:/dinov3_exports/dinov3_query_embeds",
      prefix = ""
    ),
    variant = list(
      orig = "H:/dinov3_exports/dinov3_embeds",
      query = "H:/dinov3_exports/dinov3_query_embeds",
      prefix = "_"
    )
  ),
  altitudes  = c(100, 150, 200, 300, 400),
  angles     = c(45, 90, 135, 180),     # rotated query angles
  strides    = c(NA, 2, 4, 8),          # NA == raw variant; others are subsampled/top-k
  topk_ratio = c(0.01, 0.05, 0.10),
  img_indices = c(1, 289),
  weights    = c("vitb16", "vith16+", "vitl16", "vits16", "vits16+"),
  weight_fullname = c(
    vitb16    = "dinov3_vitb16",
    `vith16+` = "dinov3_vith16plus",
    vits16    = "dinov3_vits16",
    `vits16+` = "dinov3_vits16plus",
    vitl16    = "dinov3_vitl16"
  ),
  weight_type = c(LVD = "LVD"),
  target_res = 1024
)

# --- Helpers -----------------------------------------------------------------
np <- import("numpy")
angle_tag   <- function(angle) sprintf("%03d", angle)
img_tag     <- function(idx) sprintf("%04d", as.integer(idx))
variant_tag <- function(stride = NULL, top_ratio = NULL) {
  if (is.null(stride) || is.na(stride)) return("raw")
  if (is.null(top_ratio) || is.na(top_ratio)) stop("top_ratio required when stride is provided")
  glue("sub{stride}_top{sprintf('%02d', round(top_ratio * 100))}p")
}
variant_prefix <- function(vtag) if (vtag == "raw") "raw" else glue("subsample_{vtag}")
storage_for_stride <- function(stride) if (is.na(stride)) config$roots$raw else config$roots$variant
prefix_dir <- function(name, prefix) {
  if (prefix == "" || is.na(name)) return(as.character(name))
  if (startsWith(as.character(name), prefix)) return(as.character(name))
  paste0(prefix, name)
}
first_existing_dir <- function(paths) {
  for (p in paths) {
    if (dir.exists(p)) return(p)
  }
  paths[[1]]
}

safe_np_load <- function(path) {
  if (!file.exists(path)) stop(glue("Missing file: {path}"))
  np$load(path)
}

safe_json <- function(path) {
  if (!file.exists(path)) stop(glue("Missing file: {path}"))
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

build_orig_paths <- function(weight_key, altitude, img_idx, weight_type = "LVD",
                             stride = NA, top_ratio = NA,
                             token = c("PatchToken", "PatchGrid", "GlobalToken"),
                             include_scores = TRUE) {
  token <- match.arg(token)
  vtag <- if (is.na(stride)) "raw" else variant_tag(stride, top_ratio)
  vprefix <- variant_prefix(vtag)
  wfull <- config$weight_fullname[[weight_key]]
  storage <- storage_for_stride(stride)
  other_storage <- if (identical(storage$orig, config$roots$raw$orig)) config$roots$variant else config$roots$raw
  base_dir <- first_existing_dir(c(
    file.path(storage$orig, config$dataset,
              prefix_dir(weight_key, storage$prefix),
              prefix_dir(altitude, storage$prefix),
              prefix_dir(token, storage$prefix)),
    file.path(storage$orig, config$dataset, weight_key, altitude, token),
    file.path(other_storage$orig, config$dataset,
              prefix_dir(weight_key, other_storage$prefix),
              prefix_dir(altitude, other_storage$prefix),
              prefix_dir(token, other_storage$prefix)),
    file.path(other_storage$orig, config$dataset, weight_key, altitude, token)
  ))
  stem <- glue("{token}_res{config$target_res}_{vprefix}_{wfull}_{weight_type}_{altitude}_{img_tag(img_idx)}")
  list(
    npy  = file.path(base_dir, glue("{stem}.npy")),
    meta = file.path(base_dir, glue("{stem}_meta.json")),
    scores = if (token %in% c("PatchToken", "PatchGrid") && vtag != "raw" && include_scores)
      file.path(base_dir, glue("{stem}_scores.npy")) else NA_character_,
    denseft = file.path(
      first_existing_dir(c(
        file.path(storage$orig, config$dataset,
                  prefix_dir(weight_key, storage$prefix),
                  prefix_dir(altitude, storage$prefix),
                  prefix_dir("DenseFT", storage$prefix)),
        file.path(other_storage$orig, config$dataset,
                  prefix_dir(weight_key, other_storage$prefix),
                  prefix_dir(altitude, other_storage$prefix),
                  prefix_dir("DenseFT", other_storage$prefix)),
        file.path(storage$orig, config$dataset, weight_key, altitude, "DenseFT"),
        file.path(other_storage$orig, config$dataset, weight_key, altitude, "DenseFT")
      )),
      glue("DenseFT_res{config$target_res}_{vprefix}_{wfull}_{weight_type}_{altitude}_{img_tag(img_idx)}.png")
    ),
    variant = vtag
  )
}

build_query_paths <- function(weight_key, altitude, img_idx, angle, weight_type = "LVD",
                              stride = NA, top_ratio = NA,
                              token = c("PatchToken", "PatchGrid", "GlobalToken"),
                              include_scores = TRUE) {
  token <- match.arg(token)
  vtag <- if (is.na(stride)) "raw" else variant_tag(stride, top_ratio)
  vprefix <- variant_prefix(vtag)
  wfull <- config$weight_fullname[[weight_key]]
  angle_tagged <- angle_tag(angle)
  storage <- storage_for_stride(stride)
  other_storage <- if (identical(storage$query, config$roots$raw$query)) config$roots$variant else config$roots$raw
  base_dir <- first_existing_dir(c(
    file.path(storage$query, config$dataset,
              prefix_dir(weight_key, storage$prefix),
              prefix_dir(altitude, storage$prefix),
              prefix_dir(angle_tagged, storage$prefix),
              prefix_dir(token, storage$prefix)),
    file.path(storage$query, config$dataset, weight_key, altitude, angle_tagged, token),
    file.path(other_storage$query, config$dataset,
              prefix_dir(weight_key, other_storage$prefix),
              prefix_dir(altitude, other_storage$prefix),
              prefix_dir(angle_tagged, other_storage$prefix),
              prefix_dir(token, other_storage$prefix)),
    file.path(other_storage$query, config$dataset, weight_key, altitude, angle_tagged, token)
  ))
  stem <- glue("Query{token}_res{config$target_res}_{vprefix}_{wfull}_{weight_type}_{altitude}_{img_tag(img_idx)}_rot{angle_tagged}_crop50")
  list(
    npy  = file.path(base_dir, glue("{stem}.npy")),
    meta = file.path(base_dir, glue("{stem}_meta.json")),
    scores = if (token %in% c("PatchToken", "PatchGrid") && vtag != "raw" && include_scores)
      file.path(base_dir, glue("{stem}_scores.npy")) else NA_character_,
    denseft = file.path(
      first_existing_dir(c(
        file.path(storage$query, config$dataset,
                  prefix_dir(weight_key, storage$prefix),
                  prefix_dir(altitude, storage$prefix),
                  prefix_dir(angle_tagged, storage$prefix),
                  prefix_dir("DenseFT", storage$prefix)),
        file.path(other_storage$query, config$dataset,
                  prefix_dir(weight_key, other_storage$prefix),
                  prefix_dir(altitude, other_storage$prefix),
                  prefix_dir(angle_tagged, other_storage$prefix),
                  prefix_dir("DenseFT", other_storage$prefix)),
        file.path(storage$query, config$dataset, weight_key, altitude, angle_tagged, "DenseFT"),
        file.path(other_storage$query, config$dataset, weight_key, altitude, angle_tagged, "DenseFT")
      )),
      glue("QueryDenseFT_res{config$target_res}_{vprefix}_{wfull}_{weight_type}_{altitude}_{img_tag(img_idx)}_rot{angle_tagged}_crop50.png")
    ),
    variant = vtag
  )
}

cosine_vec <- function(a, b) sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))

cosine_per_patch <- function(mat_a, mat_b) {
  if (nrow(mat_a) != nrow(mat_b) || ncol(mat_a) != ncol(mat_b)) {
    stop("Patch matrices must have the same shape for cosine similarity")
  }
  map_dbl(seq_len(nrow(mat_a)), ~ cosine_vec(mat_a[.x, ], mat_b[.x, ]))
}

select_topk_idx <- function(scores, ratio) {
  k <- max(1, floor(length(scores) * ratio))
  order(scores, decreasing = TRUE)[seq_len(k)]
}

overlap_metrics <- function(scores_a, scores_b, ratios) {
  tibble(ratio = ratios) %>%
    mutate(
      idx_a = map(ratio, ~ select_topk_idx(scores_a, .x)),
      idx_b = map(ratio, ~ select_topk_idx(scores_b, .x)),
      inter = map2_int(idx_a, idx_b, ~ length(intersect(.x, .y))),
      k_eff = map_int(idx_a, length),
      overlap_ratio = inter / k_eff,
      topk_pct = ratio * 100
    ) %>%
    select(ratio, topk_pct, overlap_ratio, k_eff = k_eff, overlap_count = inter)
}

score_correlation <- function(scores_a, scores_b, ratios) {
  tibble(scope = "all", ratio = NA_real_, corr = cor(scores_a, scores_b, use = "complete.obs")) %>%
    bind_rows(
      tibble(ratio = ratios) %>%
        mutate(
          idx_a = map(ratio, ~ select_topk_idx(scores_a, .x)),
          idx_b = map(ratio, ~ select_topk_idx(scores_b, .x)),
          corr = map2_dbl(idx_a, idx_b, ~ cor(scores_a[.x], scores_b[.y], use = "complete.obs")),
          scope = "topk",
          topk_pct = ratio * 100
        ) %>% select(scope, ratio, topk_pct, corr)
    )
}

# --- Core analysis -----------------------------------------------------------

analyze_pair <- function(weight_key, altitude, img_idx, angle, weight_type = "LVD",
                         stride = NA, top_ratio = NA,
                         token = "PatchToken", ratios = c(0.01, 0.05, 0.10)) {
  opaths <- build_orig_paths(weight_key, altitude, img_idx, weight_type, stride, top_ratio, token)
  qpaths <- build_query_paths(weight_key, altitude, img_idx, angle, weight_type, stride, top_ratio, token)

  patch_orig  <- safe_np_load(opaths$npy)
  patch_query <- safe_np_load(qpaths$npy)

  # Expect shape: [tokens, dim]
  cos_vals <- cosine_per_patch(patch_orig, patch_query)
  cos_summary <- tibble(
    weight = weight_key,
    weight_type = weight_type,
    altitude = altitude,
    angle = angle,
    variant = opaths$variant,
    img_idx = img_idx,
    cos_mean = mean(cos_vals),
    cos_sd = sd(cos_vals)
  )

  # Score overlap (if scores exist), else compute from L2 norm
  scores_orig <- if (!is.na(opaths$scores)) safe_np_load(opaths$scores) else sqrt(rowSums(patch_orig^2))
  scores_query <- if (!is.na(qpaths$scores)) safe_np_load(qpaths$scores) else sqrt(rowSums(patch_query^2))

  overlap <- overlap_metrics(scores_orig, scores_query, ratios) %>%
    mutate(weight = weight_key, weight_type = weight_type, altitude = altitude,
           angle = angle, variant = opaths$variant, img_idx = img_idx, token = token)

  score_corr <- score_correlation(scores_orig, scores_query, ratios) %>%
    mutate(weight = weight_key, weight_type = weight_type, altitude = altitude,
           angle = angle, variant = opaths$variant, img_idx = img_idx, token = token)

  list(
    cosine = cos_vals,
    cosine_summary = cos_summary,
    overlap = overlap,
    score_corr = score_corr,
    paths = list(orig = opaths, query = qpaths)
  )
}

# Build DenseFT pairing table for qualitative checks
build_denseft_pairs <- function(weight_key, altitude, img_idx, angles = c(0, config$angles),
                                stride = NA, top_ratio = NA, weight_type = "LVD") {
  base <- build_orig_paths(weight_key, altitude, img_idx, weight_type, stride, top_ratio, token = "GlobalToken")
  qlist <- map(angles[angles != 0], ~ build_query_paths(weight_key, altitude, img_idx, .x, weight_type, stride, top_ratio, token = "GlobalToken"))
  tibble(
    angle = angles,
    denseft_path = c(base$denseft, map_chr(qlist, "denseft")),
    variant = base$variant,
    weight = weight_key,
    altitude = altitude,
    img_idx = img_idx,
    weight_type = weight_type
  )
}

# Utility: rank top-N embeddings by overlap mean
rank_top_embeddings <- function(overlap_df, n = 5) {
  overlap_df %>%
    group_by(weight, weight_type, variant, stride = variant, altitude) %>%
    summarise(mean_overlap = mean(overlap_ratio), .groups = "drop") %>%
    arrange(desc(mean_overlap)) %>%
    slice_head(n = n)
}

build_eval_grid <- function(weight_keys,
                            altitudes = config$altitudes,
                            angles = config$angles,
                            strides = config$strides,
                            top_ratios = config$topk_ratio,
                            img_indices = config$img_indices) {
  sub_strides <- strides[!is.na(strides)]
  raw_grid <- expand_grid(
    weight_key = weight_keys,
    altitude = altitudes,
    angle = angles,
    img_idx = img_indices
  ) %>% mutate(stride = NA_real_, top_ratio = NA_real_)

  variant_grid <- if (length(sub_strides) > 0) {
    expand_grid(
      weight_key = weight_keys,
      altitude = altitudes,
      angle = angles,
      stride = sub_strides,
      top_ratio = top_ratios,
      img_idx = img_indices
    )
  } else {
    tibble(weight_key = character(), altitude = numeric(), angle = numeric(),
           stride = numeric(), top_ratio = numeric(), img_idx = numeric())
  }

  bind_rows(raw_grid, variant_grid)
}

analyze_weight <- function(weight_key,
                           altitudes = config$altitudes,
                           angles = config$angles,
                           strides = config$strides,
                           top_ratios = config$topk_ratio,
                           img_indices = config$img_indices,
                           token = "PatchToken") {
  grid <- build_eval_grid(weight_key, altitudes, angles, strides, top_ratios, img_indices)
  grid <- grid %>% mutate(res = pmap(
    list(weight_key, altitude, img_idx, angle, stride, top_ratio),
    ~ analyze_pair(
      weight_key = ..1,
      altitude = ..2,
      img_idx = ..3,
      angle = ..4,
      weight_type = "LVD",
      stride = ..5,
      top_ratio = ..6,
      token = token,
      ratios = config$topk_ratio
    )
  ))

  list(
    overlap = grid %>%
      transmute(weight_key, altitude, angle, stride, top_ratio, img_idx, data = map(res, "overlap")) %>%
      unnest(data),
    score_corr = grid %>%
      transmute(weight_key, altitude, angle, stride, top_ratio, img_idx, data = map(res, "score_corr")) %>%
      unnest(data),
    cosine_summary = grid %>%
      transmute(weight_key, altitude, angle, stride, top_ratio, img_idx, data = map(res, "cosine_summary")) %>%
      unnest(data)
  )
}

# --- Plot helpers ------------------------------------------------------------
plot_overlap_lines <- function(overlap_df) {
  ggplot(overlap_df, aes(x = factor(angle), y = overlap_ratio,
                        color = variant, linetype = factor(topk_pct))) +
    geom_line(aes(group = interaction(variant, topk_pct))) +
    geom_point(size = 1.5) +
    facet_grid(weight ~ .) +
    labs(x = "angle", y = "overlap_ratio", color = "variant",
         linetype = "topk(%)",
         title = "TopK overlap vs angle") +
    theme_minimal()
}

plot_cosine_hist <- function(cos_vals, angle, variant, weight) {
  tibble(cosine = cos_vals) %>%
    ggplot(aes(x = cosine)) +
    geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
    labs(title = glue("Cosine similarity angle={angle}, variant={variant}, weight={weight}"), x = "cosine", y = "count") +
    theme_minimal()
}

plot_score_corr <- function(score_corr_df) {
  ggplot(score_corr_df, aes(x = factor(angle), y = corr, color = scope)) +
    geom_line(aes(group = scope)) +
    geom_point() +
    facet_grid(weight ~ variant) +
    labs(x = "angle", y = "corr(score_orig, score_rot)", title = "Score stability") +
    theme_minimal()
}

# --- Example (guarded) -------------------------------------------------------
if (FALSE) {
  # Single example: vith16+ / alt 150 / img 1 / angle 45 / stride 2 / top 5%
  res <- analyze_pair(
    weight_key = "vith16+",
    altitude = 150,
    img_idx = 1,
    angle = 45,
    weight_type = "LVD",
    stride = 2,
    top_ratio = 0.05,
    token = "PatchToken",
    ratios = config$topk_ratio
  )

  print(res$cosine_summary)
  print(res$overlap)
  print(res$score_corr)

  # DenseFT pairing table for qualitative inspection
  df_pairs <- build_denseft_pairs("vith16+", 150, 1, angles = c(0, config$angles), stride = 2, top_ratio = 0.05)
  print(df_pairs)

  # ---------------------------------------------------------------------------
  # Per-weight batch example (limits runtime): run one weight at a time using
  # the new helper that respects the D:/ (raw) vs H:/ (subsampled) split.
  # ---------------------------------------------------------------------------
  vith16_plus_only <- analyze_weight(
    weight_key = "vith16+",
    altitudes = c(100, 150, 200, 300, 400),
    angles = config$angles,
    strides = c(NA, 2, 4, 8),
    top_ratios = config$topk_ratio,
    img_indices = c(1, 289),
    token = "PatchToken"
  )

  overlap_all <- vith16_plus_only$overlap
  scorecorr_all <- vith16_plus_only$score_corr
  cosine_summary_all <- vith16_plus_only$cosine_summary

  # To process all weights sequentially without overloading memory/time:
  # all_weights <- map(config$weights, analyze_weight)
  # names(all_weights) <- config$weights
  # bind_rows(map(all_weights, "overlap"), .id = "weight_key") -> combined_overlap

  # CSV export examples (uncomment if needed)
  # write.csv(overlap_all, "overlap_all.csv", row.names = FALSE)
  # write.csv(scorecorr_all, "scorecorr_all.csv", row.names = FALSE)
  # write.csv(cosine_summary_all, "cosine_summary_all.csv", row.names = FALSE)
}
