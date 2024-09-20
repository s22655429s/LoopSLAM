#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <eigen3/Eigen/Dense>
#include <thread>

FILE* fp_laser_odom_to_init;
FILE* fp_aft_mapped_to_init;
FILE* fp_aft_mapped_to_init_high_frec;
tf::Transform tf_velo2cam;

void WriteLaserOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
        tf::Transform odomAftMapped;
	odomAftMapped.setRotation(tf::Quaternion(odom_msg->pose.pose.orientation.x,odom_msg->pose.pose.orientation.y,
						  odom_msg->pose.pose.orientation.z,odom_msg->pose.pose.orientation.w));
	odomAftMapped.setOrigin(tf::Vector3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
					     odom_msg->pose.pose.position.z));
        tf::Transform odom_cam=tf_velo2cam*odomAftMapped*tf_velo2cam.inverse();
        tf::Matrix3x3 R= odom_cam.getBasis();
	tf::Vector3 T=odom_cam.getOrigin();
	fprintf(fp_laser_odom_to_init,"%le %le %le %le %le %le %le %le %le %le %le %le\n",
	   R[0][0],R[0][1],R[0][2],T[0],
	   R[1][0],R[1][1],R[1][2],T[1],
	   R[2][0],R[2][1],R[2][2],T[2]); 
	std::cout<<odom_msg->header.stamp<<" WriteOdom laser_odom_to_init"<<std::endl;
}
/**
* @brief  write odom file,
* @return void
*/
void WriteOdomOdomAftMapped(const nav_msgs::Odometry::ConstPtr& odom_msg)
{

        tf::Transform odomAftMapped;
	odomAftMapped.setRotation(tf::Quaternion(odom_msg->pose.pose.orientation.x,odom_msg->pose.pose.orientation.y,
						  odom_msg->pose.pose.orientation.z,odom_msg->pose.pose.orientation.w));
	odomAftMapped.setOrigin(tf::Vector3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
					     odom_msg->pose.pose.position.z));
        tf::Transform odom_cam=tf_velo2cam*odomAftMapped*tf_velo2cam.inverse();
        tf::Matrix3x3 R= odom_cam.getBasis();
	tf::Vector3 T=odom_cam.getOrigin();
	fprintf(fp_aft_mapped_to_init,"%le %le %le %le %le %le %le %le %le %le %le %le\n",
	   R[0][0],R[0][1],R[0][2],T[0],
	   R[1][0],R[1][1],R[1][2],T[1],
	   R[2][0],R[2][1],R[2][2],T[2]); 
	std::cout<<odom_msg->header.stamp<<" WriteOdom aft_mapped_to_init"<<std::endl;
}
/**
* @brief  write odom file,
* @return void
*/
void WriteOdomAftMappedHighFrec(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
        tf::Transform odomAftMapped;
	odomAftMapped.setRotation(tf::Quaternion(odom_msg->pose.pose.orientation.x,odom_msg->pose.pose.orientation.y,
						  odom_msg->pose.pose.orientation.z,odom_msg->pose.pose.orientation.w));
	odomAftMapped.setOrigin(tf::Vector3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
					     odom_msg->pose.pose.position.z));
        tf::Transform odom_cam=tf_velo2cam*odomAftMapped*tf_velo2cam.inverse();
        tf::Matrix3x3 R= odom_cam.getBasis();
	tf::Vector3 T=odom_cam.getOrigin();
	fprintf(fp_aft_mapped_to_init_high_frec,"%le %le %le %le %le %le %le %le %le %le %le %le\n",
	   R[0][0],R[0][1],R[0][2],T[0],
	   R[1][0],R[1][1],R[1][2],T[1],
	   R[2][0],R[2][1],R[2][2],T[2]); 
	std::cout<<odom_msg->header.stamp<<" WriteOdom aft_mapped_to_init_high_frec"<<std::endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_helper");
    ros::NodeHandle n("~");
    std::string dataset_folder,sequence_number, output_path,output_laser_odom_to_init,output_aft_mapped_to_init,output_aft_mapped_to_init_high_frec;
    n.getParam("output_path", output_path);
    
    output_laser_odom_to_init=output_path+"/laser_odom_to_init.txt";
    fp_laser_odom_to_init = fopen(output_laser_odom_to_init.c_str(),"w");
    
    output_aft_mapped_to_init=output_path+"/aft_mapped_to_init.txt";
    fp_aft_mapped_to_init = fopen(output_aft_mapped_to_init.c_str(),"w");
    
    output_aft_mapped_to_init_high_frec=output_path+"/aft_mapped_to_init_high_frec.txt";
    fp_aft_mapped_to_init_high_frec = fopen(output_aft_mapped_to_init_high_frec.c_str(),"w");
  
    std::string calib_file = n.param<std::string>("calib_file"," ");
    std::cout<<"calib_file= "<<calib_file<<std::endl;
    std::ifstream fin ( calib_file );
    std::string tmp;
    for(int i=0;i<4;i++) std::getline(fin,tmp);
    Eigen::Matrix4d matrix_tmp;
    matrix_tmp.setIdentity();
    fin>>tmp>>matrix_tmp(0,0)>>matrix_tmp(0,1)>>matrix_tmp(0,2)>>matrix_tmp(0,3)
    >>matrix_tmp(1,0)>>matrix_tmp(1,1)>>matrix_tmp(1,2)>>matrix_tmp(1,3)
    >>matrix_tmp(2,0)>>matrix_tmp(2,1)>>matrix_tmp(2,2)>>matrix_tmp(2,3);
    
    Eigen::Vector3d eulerAngle = matrix_tmp.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
    std::cout<<"eulerAngle="<<eulerAngle* (180 / M_PI)<<matrix_tmp<<std::endl;  
    /*Eigen::Matrix3d rotation_matrix2;
    rotation_matrix2 = Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitZ()) * 
                       Eigen::AngleAxisd(eulerAngle[1]-0.5*M_PI/180, Eigen::Vector3d::UnitY()) * 
                       Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitX());
    std::cout<<"eulerAngle="<<rotation_matrix2.eulerAngles(2, 1, 0)* (180 / M_PI)<<std::endl;
    matrix_tmp.block<3, 3>(0, 0)=rotation_matrix2;
    printf("%.12le %.12le %.12le %.12le %.12le %.12le %.12le %.12le %.12le %.12le %.12le %.12le\n",
                   matrix_tmp(0,0), matrix_tmp(0,1), matrix_tmp(0,2), matrix_tmp(0,3),
                   matrix_tmp(1,0), matrix_tmp(1,1), matrix_tmp(1,2), matrix_tmp(1,3),
                   matrix_tmp(2,0), matrix_tmp(2,1), matrix_tmp(2,2), matrix_tmp(2,3));*/

    tf_velo2cam.setBasis(tf::Matrix3x3(matrix_tmp(0,0),matrix_tmp(0,1),matrix_tmp(0,2),
				       matrix_tmp(1,0),matrix_tmp(1,1),matrix_tmp(1,2),
				       matrix_tmp(2,0),matrix_tmp(2,1),matrix_tmp(2,2)));
    tf_velo2cam.setOrigin(tf::Vector3(matrix_tmp(0,3),matrix_tmp(1,3),matrix_tmp(2,3)));
	
    //ros::Subscriber subLaserOdometry = n.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 1024, WriteLaserOdometry);
    ros::Subscriber subLaserOdometry = n.subscribe<nav_msgs::Odometry>("/odom", 1024, WriteLaserOdometry);
    ros::Subscriber subOdomAftMapped = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 1024, WriteOdomOdomAftMapped );
    ros::Subscriber subOdomAftMappedHighFrec = n.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 1024, WriteOdomAftMappedHighFrec);
    
    //std::thread publish_data_process{pub_data_process};
    ros::spin();
}