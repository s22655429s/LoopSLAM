#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <global_graph/keyframe.hpp>
#include <global_graph/registrations.hpp>
#include <global_graph/graph_slam.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

#include <DBoW3/DBoW3.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;

namespace slam
{

  struct Loop
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Loop>;

    Loop(const KeyFrame::Ptr &key1, const KeyFrame::Ptr &key2, const Eigen::Matrix4f &relpose)  //Loop結構的構造函數，它接受兩個關鍵幀指針和一個相對位姿矩陣作為參數
        : key1(key1),
          key2(key2),
          relative_pose(relpose)
    {
    }

  public:
    KeyFrame::Ptr key1;
    KeyFrame::Ptr key2;
    Eigen::Matrix4f relative_pose;
  };

  /**
 * @brief this class finds loops by scam matching and adds them to the pose graph
 */
  class LoopDetector
  {
  public:
    typedef pcl::PointXYZI PointT;

    /**
   * @brief constructor
   * @param pnh
   */
    LoopDetector(ros::NodeHandle &pnh)
    {
      distance_thresh = pnh.param<double>("distance_thresh", 5.0);
      accum_distance_thresh = pnh.param<double>("accum_distance_thresh", 8.0);
      distance_from_last_edge_thresh = pnh.param<double>("min_edge_interval", 5.0);

      fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
      fitness_score_thresh = pnh.param<double>("fitness_score_thresh", 0.5);

      registration = select_registration_method(pnh);
      last_edge_accum_distance = 0.0;

      //loop dectection param
      voc_path = pnh.param<std::string>("voc_path", "$(find slam)/config/vocab_larger.yml.gz");
      voc = new DBoW3::Vocabulary(voc_path);
      if (voc->empty())
      {
        std::cout << "Vocabulary dose not exit." << std::endl;
        return;
      }
    }

    /**
   * @brief detect loops and add them to the pose graph
   * @param keyframes       keyframes
   * @param new_keyframes   newly registered keyframes
   * @param graph_slam      pose graph
   */
    std::vector<Loop::Ptr> detect(const std::vector<KeyFrame::Ptr> &keyframes, const std::deque<KeyFrame::Ptr> &new_keyframes, slam::GraphSLAM &graph_slam)
    {
      std::vector<Loop::Ptr> detected_loops;  //初始化一個向量來存儲檢測到的迴圈
      for (const auto &new_keyframe : new_keyframes)  //查看每一個新註冊的關鍵幀
      {
        auto candidates = find_candidates(keyframes, new_keyframe);  //從keyframes裡找跟new_keyframe的疑似閉回合路幀,candidates是一個容器,包含多個疑似閉回合路幀
        auto loop = matching_and_bow(candidates, new_keyframe, graph_slam);
        if (loop)
        {
          detected_loops.push_back(loop);
        }
      }

      return detected_loops;
    }

    double get_distance_thresh() const
    {
      return distance_thresh;
    }

  private:
    /**
   * @brief find loop candidates. A detected loop begins at one of #keyframes and ends at #new_keyframe
   * @param keyframes      candidate keyframes of loop start
   * @param new_keyframe   loop end keyframe
   * @return loop candidates
   */
    std::vector<KeyFrame::Ptr> find_candidates(const std::vector<KeyFrame::Ptr> &keyframes, const KeyFrame::Ptr &new_keyframe) const
    {
      // too close to the last registered loop edge
      if (new_keyframe->accum_distance - last_edge_accum_distance < distance_from_last_edge_thresh)  //如果新關鍵幀與最後一次註冊的迴圈邊的累計距離小於設定的閾值
      {
        return std::vector<KeyFrame::Ptr>();  //函數將返回一個空的候選列表
      }

      std::vector<KeyFrame::Ptr> candidates;  //初始化一個用來儲存候選關鍵幀的向量
      candidates.reserve(32); //预留空间

      for (const auto &k : keyframes)  //查看keyframes 向量中的每一個關鍵幀
      {
        // traveled distance between keyframes is too small
        if (new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh)  //如果新關鍵幀與當前關鍵幀之間的累計距離小於閾值（accum_distance_thresh），則跳過當前關鍵幀。這確保了只有在兩個關鍵幀之間有足夠的距離時，才考慮它們形成迴圈。
        {
          continue;
        }

        const auto &pos1 = k->node->estimate().translation();  //
        const auto &pos2 = new_keyframe->node->estimate().translation();

        // estimated distance between keyframes is too small
        double dist = (pos1.head<2>() - pos2.head<2>()).norm();
        if (dist > distance_thresh)
        {
          continue;
        }

        candidates.push_back(k);
      }

      return candidates;
    }

    /**
   * @brief To validate a loop candidate this function applies a scan matching between keyframes consisting the loop. If they are matched well, the loop is added to the pose graph
   * @param candidate_keyframes  candidate keyframes of loop start
   * @param new_keyframe         loop end keyframe
   * @param graph_slam           graph slam
   */
    Loop::Ptr matching(const std::vector<KeyFrame::Ptr> &candidate_keyframes, const KeyFrame::Ptr &new_keyframe, slam::GraphSLAM &graph_slam)
    {
      if (candidate_keyframes.empty())
      {
        return nullptr;
      }

      registration->setInputTarget(new_keyframe->cloud);

      double best_score = std::numeric_limits<double>::max();
      KeyFrame::Ptr best_matched;
      Eigen::Matrix4f relative_pose;

      std::cout << std::endl;
      std::cout << "--- loop detection ---" << std::endl;
      std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
      std::cout << "matching" << std::flush;
      auto t1 = ros::Time::now();

      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      for (const auto &candidate : candidate_keyframes)
      {
        registration->setInputSource(candidate->cloud);
        Eigen::Matrix4f guess = (new_keyframe->node->estimate().inverse() * candidate->node->estimate()).matrix().cast<float>();
        guess(2, 3) = 0.0;
        registration->align(*aligned, guess);
        std::cout << "." << std::flush;

        double score = registration->getFitnessScore(fitness_score_max_range);
        if (!registration->hasConverged() || score > best_score)
        {
          continue;
        }

        best_score = score; //取最低匹配误差（得分）
        best_matched = candidate;
        relative_pose = registration->getFinalTransformation();
      }

      auto t2 = ros::Time::now();
      std::cout << " done" << std::endl;
      std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

      if (best_score > fitness_score_thresh)
      {
        std::cout << "loop not found..." << std::endl;
        return nullptr;
      }

      std::cout << "loop found!!" << std::endl; //block 表示返回从矩阵的(i, j)开始，每行取p个元素，每列取q个元素
      std::cout << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - " << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << std::endl;

      last_edge_accum_distance = new_keyframe->accum_distance;

      return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
    }

    /**
   * @brief To validate a loop candidate this function applies img bow loop detection and a scan matching between keyframes. If true, the loop is added to the pose graph
   * @param candidate_keyframes  candidate keyframes of loop start
   * @param new_keyframe         loop end keyframe
   * @param graph_slam           graph slam
   */
    Loop::Ptr matching_and_bow(const std::vector<KeyFrame::Ptr> &candidate_keyframes, const KeyFrame::Ptr &new_keyframe, slam::GraphSLAM &graph_slam)
    {
      if (candidate_keyframes.empty())
      {
        std::cout << "no candidate, then loop not found..." << std::endl;
        return nullptr;
      }

      registration->setInputTarget(new_keyframe->cloud);  //使用新關鍵幀的點雲作為批配目標點雲

      double best_score = std::numeric_limits<double>::max();
      KeyFrame::Ptr best_matched;  //宣告儲存批配最好之關鍵幀
      Eigen::Matrix4f relative_pose;  //宣告儲存批配最好之關鍵幀與新關鍵幀之相對pose

      std::cout << std::endl;
      std::cout << "--- loop detection ---" << std::endl;
      std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
      auto t1 = ros::Time::now();  //計算必迴路檢測所需之時間

      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());  //創建一個點雲儲存對齊批配後的點雲
      DBoW3::Database db_temp;  //初始化一個DBoW3 的數據庫對象,設置其詞彙表
      db_temp.setVocabulary(*voc, false, 0);  //voc指向一個詞彙表,false代表不複製詞彙表數據,直接使用,0代表從最底層開始使用
      for (const auto &candidate : candidate_keyframes)  //對每個candidate_keyframes添加描述符
      {
        db_temp.add(candidate->descriptor);
      }
      DBoW3::QueryResults ret;  //儲存資料庫查詢的結果
      db_temp.query(new_keyframe->descriptor, ret, 5);  //找出5個與new_keyframe最相似的關鍵幀
      std::cout << candidate_keyframes.size() << "loop dectection for " << new_keyframe->seq << std::endl
                << ret << std::endl;

      for (int i = 0; i < ret.size(); i++)
      {
        if (ret[i] < 0.04 || best_score <= fitness_score_thresh)  //if best_score 已经小于或等于这个阈值,表示已經找到非常好匹配結果,可以提前中止循環,best_score 越低代表批配結果越好
          break;
        std::cout << ret[i] << ret[i].Id << std::endl;
        int best_id = ret[i].Id;  
        best_matched = candidate_keyframes[best_id];  //從candidate_keyframes 獲取對應的關鍵幀
        registration->setInputSource(best_matched->cloud);  //使用批配最好的點雲作為批配目標點雲
        Eigen::Matrix4f guess = (new_keyframe->node->estimate().inverse() * best_matched->node->estimate()).matrix().cast<float>();  //計算兩位姿之間的變換
        guess(2, 3) = 0.0;  //提供一個合理的變換猜測幫助批配算法更塊收斂
        registration->align(*aligned, guess);  //使用提供的初始猜測guess和來源點雲best_matched->cloud，嘗試將其與新關鍵影格的點雲（通常作為目標點雲）對齊
        std::cout << "." << std::flush;

        double score = registration->getFitnessScore(fitness_score_max_range);  //透過源點雲到目標點雲的平均距離評估是否對齊
        if (!registration->hasConverged() || score > best_score)   //對齊要收斂且對齊後的分數要大於對齊前 ,否則配準失敗
        {
          continue;
        }

        best_score = score; //取最低匹配误差（得分）
        relative_pose = registration->getFinalTransformation();  //保存對齊後的變換矩陣
      }

      if (best_score > fitness_score_thresh)  //表示批配不到
      {
        std::cout << "loop not found..." << std::endl;
        return nullptr;
      }

      auto t2 = ros::Time::now();
      std::cout << " done" << std::endl;
      std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

      std::cout << "loop found!!" << std::endl; //block 表示返回从矩阵的(i, j)开始，每行取p个元素，每列取q个元素
      std::cout << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - " << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << std::endl;  //輸出相對位姿訊息

      last_edge_accum_distance = new_keyframe->accum_distance;  //更新用於記錄上次環路偵測後的關鍵影格累積距離。 這個值用於未來環路檢測的初始距離判斷，避免頻繁檢測

      return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose); //建立一個新的 Loop 對象，並透過智慧指針返回。 這個物件包含了構成環路的新關鍵影格、最佳匹配關鍵影格和它們之間的相對位姿。
    }

  private:
    double distance_thresh;                // estimated distance between keyframes consisting a loop must be less than this distance
    double accum_distance_thresh;          // traveled distance between ...
    double distance_from_last_edge_thresh; // a new loop edge must far from the last one at least this distance

    double fitness_score_max_range; // maximum allowable distance between corresponding points
    double fitness_score_thresh;    // threshold for scan matching

    double last_edge_accum_distance;

    pcl::Registration<PointT, PointT>::Ptr registration;

    DBoW3::Database db;
    DBoW3::Vocabulary *voc;
    std::string voc_path;
  };
}

#endif // LOOP_DETECTOR_HPP
