#include <vector>
#include "track/TrackKLT.h"
#include "logger/Print.h"
#include "basic/CVLambdaBody.h"
#include "core/SensorData.h"

void TrackKLT::feed_new_camera(const CameraData &message) {
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]: - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]: - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }
  feed_stereo(message, 0, 1);
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);

  // 初始帧 或 重新计算特征点
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;

    // 特征检测
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, 
        cam_id_left, cam_id_right, good_left, good_right, good_ids_left, good_ids_right);

    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }

  // 上一帧数据
  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old  = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old  = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];

  // 特征检测
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
  
  // 当前特征初始化
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
    for (int i = range.start; i < range.end; i++) {
      bool is_left = (i == 0);
      perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                        is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                        is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                        is_left ? mask_ll : mask_rr);
    }
  }));

  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points\n" RESET);
    return;
  }                

  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // 左目特征点
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // 过滤边界外的点
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || 
      (int)pts_left_new.at(i).pt.x > img_left.cols || (int)pts_left_new.at(i).pt.y > img_left.rows) {
      continue;
    }
  
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }

    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // 过滤边界外的点
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows) {
        continue;
      }
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
    }
  }

  // 右目特征点
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // 过滤边界外的点
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || 
        (int)pts_right_new.at(i).pt.x >= img_right.cols || (int)pts_right_new.at(i).pt.y >= img_right.rows) {
      continue;
    }
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
    }
  }

  // 出来特征点
  for (size_t i = 0; i < good_left.size(); i++) {
    // TODO
 
  }
  for (size_t i = 0; i < good_right.size(); i++) {
  }

  // 保存当前数据
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {


  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }

    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }

    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height) {
        continue;
      }

      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127) {
        continue;
      }
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

  //   // TODO: Project points from the left frame into the right frame
  //   // TODO: This will not work for large baseline systems.....
  //   // TODO: If we had some depth estimates we could do a better projection
  //   // TODO: Or project and search along the epipolar line??
  //   std::vector<cv::KeyPoint> kpts1_new;
  //   std::vector<cv::Point2f> pts1_new;
  //   kpts1_new = kpts0_new;
  //   pts1_new = pts0_new;

  //   // If we have points, do KLT tracking to get the valid projections into the right image
  //   if (!pts0_new.empty()) {

  //     // Do our KLT tracking from the left to the right frame of reference
  //     // NOTE: we have a pretty big window size here since our projection might be bad
  //     // NOTE: but this might cause failure in cases of repeated textures (eg. checkerboard)
  //     std::vector<uchar> mask;
  //     // perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
  //     std::vector<float> error;
  //     cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  //     cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
  //                              cv::OPTFLOW_USE_INITIAL_FLOW);

  //     // Loop through and record only ones that are valid
  //     for (size_t i = 0; i < pts0_new.size(); i++) {

  //       // Check to see if the feature is out of bounds (oob) in either image
  //       bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
  //                        (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
  //       bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
  //                         (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

  //       // Check to see if it there is already a feature in the right image at this location
  //       //  1) If this is not already in the right image, then we should treat it as a stereo
  //       //  2) Otherwise we will treat this as just a monocular track of the feature
  //       // TODO: we should check to see if we can combine this new feature and the one in the right
  //       // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
  //       if (!oob_left && !oob_right && mask[i] == 1) {
  //         // update the uv coordinates
  //         kpts0_new.at(i).pt = pts0_new.at(i);
  //         kpts1_new.at(i).pt = pts1_new.at(i);
  //         // append the new uv coordinate
  //         pts0.push_back(kpts0_new.at(i));
  //         pts1.push_back(kpts1_new.at(i));
  //         // move id forward and append this new point
  //         size_t temp = ++currid;
  //         ids0.push_back(temp);
  //         ids1.push_back(temp);
  //       } else if (!oob_left) {
  //         // update the uv coordinates
  //         kpts0_new.at(i).pt = pts0_new.at(i);
  //         // append the new uv coordinate
  //         pts0.push_back(kpts0_new.at(i));
  //         // move id forward and append this new point
  //         size_t temp = ++currid;
  //         ids0.push_back(temp);
  //       }
  //     }
  //   }
  // }

  // // RIGHT: Now summarise the number of tracks in the right image
  // // RIGHT: We will try to extract some monocular features if we have the room
  // // RIGHT: This will also remove features if there are multiple in the same location
  // cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  // cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  // float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  // float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  // cv::Size size_grid1(grid_x, grid_y); // width x height
  // cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  // cv::Mat mask1_updated = mask0.clone();
  // it0 = pts1.begin();
  // it1 = ids1.begin();
  // while (it0 != pts1.end()) {
  //   // Get current left keypoint, check that it is in bounds
  //   cv::KeyPoint kpt = *it0;
  //   int x = (int)kpt.pt.x;
  //   int y = (int)kpt.pt.y;
  //   int edge = 10;
  //   if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
  //     it0 = pts1.erase(it0);
  //     it1 = ids1.erase(it1);
  //     continue;
  //   }
  //   // Calculate mask coordinates for close points
  //   int x_close = (int)(kpt.pt.x / (float)min_px_dist);
  //   int y_close = (int)(kpt.pt.y / (float)min_px_dist);
  //   if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
  //     it0 = pts1.erase(it0);
  //     it1 = ids1.erase(it1);
  //     continue;
  //   }
  //   // Calculate what grid cell this feature is in
  //   int x_grid = std::floor(kpt.pt.x / size_x1);
  //   int y_grid = std::floor(kpt.pt.y / size_y1);
  //   if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
  //     it0 = pts1.erase(it0);
  //     it1 = ids1.erase(it1);
  //     continue;
  //   }
  //   // Check if this keypoint is near another point
  //   // NOTE: if it is *not* a stereo point, then we will not delete the feature
  //   // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
  //   bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
  //   if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
  //     it0 = pts1.erase(it0);
  //     it1 = ids1.erase(it1);
  //     continue;
  //   }
  //   // Now check if it is in a mask area or not
  //   // NOTE: mask has max value of 255 (white) if it should be
  //   if (mask1.at<uint8_t>(y, x) > 127) {
  //     it0 = pts1.erase(it0);
  //     it1 = ids1.erase(it1);
  //     continue;
  //   }
  //   // Else we are good, move forward to the next point
  //   grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
  //   if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
  //     grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
  //   }
  //   // Append this to the local mask of the image
  //   if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
  //     cv::Point pt1(x - min_px_dist, y - min_px_dist);
  //     cv::Point pt2(x + min_px_dist, y + min_px_dist);
  //     cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
  //   }
  //   it0++;
  //   it1++;
  // }

  // // RIGHT: if we need features we should extract them in the current frame
  // // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  // int num_featsneeded_1 = num_features - (int)pts1.size();
  // if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

  //   // This is old extraction code that would extract from the whole image
  //   // This can be slow as this will recompute extractions for grid areas that we have max features already
  //   // std::vector<cv::KeyPoint> pts1_ext;
  //   // Grider_FAST::perform_griding(img1pyr.at(0), mask1_updated, pts1_ext, num_features, grid_x, grid_y, threshold, true);

  //   // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  //   cv::Mat mask1_grid;
  //   cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

  //   // Create grids we need to extract from and then extract our features (use fast with griding)
  //   int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  //   int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  //   std::vector<std::pair<int, int>> valid_locs;
  //   for (int x = 0; x < grid_2d_grid1.cols; x++) {
  //     for (int y = 0; y < grid_2d_grid1.rows; y++) {
  //       if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
  //         valid_locs.emplace_back(x, y);
  //       }
  //     }
  //   }
  //   std::vector<cv::KeyPoint> pts1_ext;
  //   Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

  //   // Now, reject features that are close a current feature
  //   for (auto &kpt : pts1_ext) {
  //     // Check that it is in bounds
  //     int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
  //     int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
  //     if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
  //       continue;
  //     // See if there is a point at this location
  //     if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
  //       continue;
  //     // Else lets add it!
  //     pts1.push_back(kpt);
  //     size_t temp = ++currid;
  //     ids1.push_back(temp);
  //     grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
  //   }
  // }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out) {

  runtime_assert(kpts0.size() == kpts1.size(), "kpts 0 1 is not equal\n")
  if (kpts0.empty() || kpts1.empty()) {
    return;
  }

  // 数据转换用于去外点
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // 特征点不够 ransac
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++) {
      mask_out.push_back((uchar)0);
    }
    return;
  }

  std::vector<float> error;
  std::vector<uchar> mask_klt;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // 归一化坐标，去畸变
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // 去外点
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  // 前后帧跟踪的特征点和内点的交集
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // 过滤后的特征点回传
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}
