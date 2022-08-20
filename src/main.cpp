#include <omp.h>    //并行环境变量
#include <mutex>    //线程锁
#include <math.h>
#include <thread>   //多线程
#include <fstream>  //文件输出
#include <csignal>  //进程信号处理，如ctrl+c
#include <unistd.h> //unix的std库。许多在Linux下开发的C程序都需要头文件unistd.h，但VC中没有个头文件， 所以用VC编译总是报错。

#include <ros/ros.h>
#include <Eigen/Core> 
#include <livox_ros_driver/CustomMsg.h>  

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "imuprocess.h"
#include "preprocess.h"
#include "ikd-Tree/ikd_Tree.h"
#include "comman_lib.h"


#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.00015)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)
#define NUM_MATCH_POINTS    (5)
using namespace std;

mutex mtx_buffer;   //定义互斥量
condition_variable sig_buffer;

bool time_sync_en = false;  //是否需要用代码时间同步 标志
bool lidar_pushed = false;

std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;      //存20ms的所有帧imu数据
std::deque<PointCloudXYZI::Ptr>        lidar_buffer;    //存20ms的点云数据（pcl格式）
std::deque<double>                     time_buffer;     //存每20ms中的点云的最后时间
double last_timestamp_imu = -0.01, last_timestamp_lidar = 0;
double lidar_end_time = 0, lidar_mean_scantime = 0;
int scan_num = 0;
double first_lidar_time = 0;
bool flg_first_scan = true;

MeasureGroup measurement;                   //创建对象 全局变量 -- 测量量（imu）
StateGroup g_state;                         //创建状态 全局变量 -- 系统状态（imu）//构造函数里自动赋了初置
shared_ptr<ImuProcess> p_imu(new ImuProcess());
shared_ptr<PreProcess> p_pre(new PreProcess());
PointCloudXYZI::Ptr pointcloud_undistort(new PointCloudXYZI());  //点云指针类型
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());       //滤波后的局部点云(当前20ms)
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());      //
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());         //装能成功计算残差的一部分点云(雷达坐标系)
PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());              //用来存储最近点的信息(最近点的xyz，最近点距离基准点的残差)

//---about map
vector<BoxPointType> cub_needrm;    //待清除点的盒子（ikdtree中）
vector<PointVector> Nearest_Points;
KD_TREE<PointNomalType> ikdtree;
Eigen::Vector3f XAxisPoint_body(2, 0.0, 0.0);    //x轴点-局部坐标系下    （2，0，0）
Eigen::Vector3f XAxisPoint_world(2, 0.0, 0.0);   //x轴点-世界坐标系下    （2，0，0）
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double cube_len = 1000;  //地图的局部区域长度
const float MOV_THRESHOLD = 1.5f; //移动阈值        
float DET_RANGE = 100.0f;         //雷达最大探测距离 -- avia450m，velodyne100m
pcl::VoxelGrid<PointNomalType> downSizeFilterSurf;   //创建体素滤波器    //x.setInputCloud(cloud) 输入点云； x.setLeafSize(0.5f,0.5f,0.5f)设置体素； x.filter(cloud_)过滤后的点云
pcl::VoxelGrid<PointNomalType> downSizeFilterMap;    //此滤波器针对整个地图； 上一个滤波器针对去畸变后的当前帧点云
double filter_size_surf_min = 0.5,  filter_size_map_min = 0.5; //体素滤波参数   1、点云降采样体素参数   2、地图降采样体素参数
int feats_down_size = 0;
bool initialized_map_kdtree_flag = false;

//---about ESIKF
Eigen::Vector3d pos_lid;
bool flg_EKF_converged = false;
double deltaT, deltaR;
int NUM_MAX_ITERATIONS = 3;
double m_maximum_res_dis = 0.3;//(改)
int    effct_feat_num = 0;
bool flg_EKF_inited = false;
Eigen::Matrix< double, DIM_STATE, DIM_STATE > G, H_T_H, I_STATE;   
//double total_distance = 0.0;
//Eigen::Vector3d position_last = Eigen::Vector3d::Zero();
int iterCount = 0;

//---about ros pub/sub
nav_msgs::Path IMU_path;                    //IMU path 用于发布
ros::Publisher pub_pre_odometry;            //imu位姿先验的发布者
ros::Publisher pub_pre_path;                //
ros::Publisher pub_pre_pointcloud;
ros::Publisher pub_pointcloud;
ros::Publisher pubLaserCloudFull;           //发布全局点云地图
double timediff_lidar_and_imu = 0.0;        //imu和雷达偏移了多少,回调中处理ros调整时间
bool timediff_set_flg = false;              //是否需时间同步---imu雷达回调配合使用
bool dense_pub_en = true;                   //发布点云是否少发布一些点

//---
Eigen::Vector3d Lidar_T_wrt_IMU(Eigen::Vector3d::Zero());
Eigen::Matrix3d Lidar_R_wrt_IMU(Eigen::Matrix3d::Identity());
std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);

void velodyne_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    if(msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        cout << "velodyne雷达时间戳不对---clear buffer of lidar" << endl;
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

//多个话题的回调是按顺序串行执行的
void livox_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();           //-----------------------lock
    if(msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        cout << "livox雷达时间戳不对---clear buffer of lidar" << endl;
        lidar_buffer.clear();       //pcl格式
    }
    last_timestamp_lidar = msg->header.stamp.toSec();   //首个点云的时间    //更新时间戳
    if(!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())  //缓存器有数据,但时间不同步 且时间偏移大于10
    {
        cout << "imu and lidar have big gap on time, imu header time:" << last_timestamp_imu << ", lidar header time:" << last_timestamp_lidar << endl;
    }
    if(time_sync_en && !timediff_set_flg && abs(last_timestamp_imu - last_timestamp_lidar) > 1 && !imu_buffer.empty())  //需要时间同步且有偏移,这里面只会走一次
    {   
        timediff_set_flg = true;
        timediff_lidar_and_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;   //计算两传感器偏移时间
        cout << "自动对齐时间,时间偏移" << timediff_lidar_and_imu <<endl;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());  //定义一个新的点云，用ptr指向他
    p_pre->process(msg, ptr);    //参数1：待转换的livox点云， 参数2：转换后pcl点云格式 的容器
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar); 

    mtx_buffer.unlock();         //-----------------------unlock
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));   //堆区开辟空间给msg_in，用msg指向他（简单来说就是拷贝，因为用const接收）
    //先判断是否需要纠正时间的偏移
    if(time_sync_en && abs(timediff_lidar_and_imu) > 0.1)   //需要时间同步且时间偏移大于0.1,调整时间戳
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_and_imu + msg_in->header.stamp.toSec());    //将浮点型格式 转化为 时间戳格式
    }

    double imu_timestamp = msg->header.stamp.toSec();   //20ms中imu第一帧的时间
    mtx_buffer.lock();          //-----------------------lock
    if(imu_timestamp < last_timestamp_imu)
    {
        cout << "imu时间戳不对---clear buffer of imu" << endl;
        imu_buffer.clear();
    }
    last_timestamp_imu = imu_timestamp; //时间戳传递
    //cout << "in cbk: msg = \n" << msg->linear_acceleration << endl;   //可以用来确定imu坐标系
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();        //-----------------------unlock
    sig_buffer.notify_all();
}

bool buffer_to_meas(MeasureGroup &meas) //把buffer数据拿到meas中
{
    if(imu_buffer.empty() || lidar_buffer.empty())
    {
        //cout << "imu_buffer = 0" << endl;
        return false;
    }

    if(lidar_pushed == false)   //目的就是确定雷达的开始时间和结束时间
    {
        meas.lidar = lidar_buffer.front();  //把缓存器中第一个20ms的点云给到meas
        meas.lidar_beg_time = time_buffer.front();  //把时间也给了                  //lidar 开始时间赋值完毕------------
        
        //设定20ms中的最后点时间（需要分情况，不是最后点是多少时间就是多少）    //目的就是求 lidar_end_time
        if(meas.lidar->points.size() <= 1)
        {
            cout << "meas has too little points" << endl;
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;   //结束时间 = 这个点+雷达的平均扫描时间
        }
        else if(meas.lidar->points.back().curvature / double(1000) < 0.5*lidar_mean_scantime)    //meas由buffer给的，buffer由preprocess给的，preprocess中做的格式转换
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;   //curvature是该点距离20ms首点的偏移时间
        }
        else
        {
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  //最后时间和偏移时间要搞清楚
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime)/scan_num;//更新平均扫描时间
        }
        meas.lidar_end_time = lidar_end_time;                                   //lidar 结束时间赋值完毕------------
        lidar_pushed = true;
    }

    if(last_timestamp_imu < lidar_end_time) //一定是imu时间包住雷达时间，因为还要做反向传播呢   //内部imu频率200hz
    {
        return false;
    }

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear(); 
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) //若imu_buffer不为空，则把buffer中的东西转到meas中
    {
        //imu_time = imu_buffer.front()->header.stamp.toSec();  
        //if(imu_time > lidar_end_time) {break;}      //imu_time是 20ms中imu的最开始时间，last_timestamp_imu是20ms中imu的最后时间
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front(); //deque容器只能对头尾进行操作
    }

    //更新缓存器
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void pointBodyToWorld( PointNomalType const *const pi, PointNomalType *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_state.Rot * ( g_state.R_L_I * p_body + g_state.T_L_I ) + g_state.Pos );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;
}
void pointBodyToWorld(const Eigen::Vector3f &pi, Eigen::Vector3f &po) //模板函数（只转一个点，转坐标系原点？）
{
    Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
    Eigen::Vector3d p_global(g_state.Rot * (p_body + g_state.T_L_I) + g_state.Pos);
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history); //获得被删除的点云
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;           //局部地图盒子的正方体两个顶点(两顶点直接约束住局部地图整个立方体)
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();                 //清除 待清除的盒子
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    Eigen::Vector3d pos_LiD = pos_lid;
    if(!Localmap_Initialized)           //若局部地图没初始化
    {
        for(int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len/2.0;  //cube_len是地图的局部区域的长度，设定为200
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len/2.0;  //初始位姿 + 200（200用立方体个数度量？）---这里可以打印看看
        }
        Localmap_Initialized = true;    //局部地图初始化————只是设定了局部地图的最大顶点和最小顶点***
        return;
    }
    float dist_to_map_edge[3][2];       //到地图边缘的距离
    bool need_move = false;             //默认不需要移动局部地图
    for(int i = 0; i < 3; i++)          //分别计算当前位姿距离地图边缘最大点和最小点的距离
    {   
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);  //第1列向量：当前离地图最小顶点有多远
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);  //第2列向量：当前离地图最大顶点有多远
        if(dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE  ||  dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            need_move = true;           //当前位姿离地图边缘的距离<1.5倍探测距离(任意一轴)，则需要移动局部地图
        }
    }
    if(need_move == false)              //若不需要移动地图，就退出此函数
    {return;}

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for(int i = 0; i < 3; i++)          //分别处理三个轴
    {
        tmp_boxpoints = LocalMap_Points;
        if(dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) //若x轴上 距离地图边界的距离小于1.5倍的探测距离，且离最小顶点近
        {
            New_LocalMap_Points.vertex_max[i] = New_LocalMap_Points.vertex_max[i] - mov_dist;
            New_LocalMap_Points.vertex_min[i] = New_LocalMap_Points.vertex_min[i] - mov_dist;
            tmp_boxpoints.vertex_min[i] =  LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)    //离最大顶点近了
        {
            New_LocalMap_Points.vertex_max[i] = New_LocalMap_Points.vertex_max[i] + mov_dist;
            New_LocalMap_Points.vertex_min[i] = New_LocalMap_Points.vertex_min[i] + mov_dist;
            tmp_boxpoints.vertex_max[i] =  LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;  //更新移动后的局部地图极限顶点
    points_cache_collect();
    if(cub_needrm.size() > 0)
    {
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); //清除盒内的点
    }
}

void initialize_map_kdtree(int feats_down_size)
{
    initialized_map_kdtree_flag = false;
    if(ikdtree.Root_Node == nullptr)
    {
        if(feats_down_size > 5)
        {
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);          //世界坐标系下降采样点的空间大小重置--初始化
            for (int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
            initialized_map_kdtree_flag = true;
        }
    }
}

void set_initial_state_cov( StateGroup &state )
{
    // Eigen::MatrixXd H_init(Eigen::Matrix<double, 15, DIM_STATE>::Zero());
    // Eigen::MatrixXd z_init(Eigen::Matrix<double, 15, 1>::Zero());
    // H_init.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    // H_init.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
    // H_init.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    // H_init.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

    state.cov = state.cov.setIdentity() * INIT_COV;
    state.cov.block( 0, 0, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // R
    state.cov.block( 3, 3, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // T
    state.cov.block( 6, 6, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // vel
    state.cov.block( 9, 9, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-3;   // bias_g
    state.cov.block( 12, 12, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-1; // bias_a
    state.cov.block( 15, 15, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5; // Gravity   
}

template<typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    /*  
        ax + by + cz + d = 0; 若d != 0,则两边除以d，平面：a/d x + b/d y + c/d z + 1 = 0, 重新记a/d = pa, b/d = pb, c/d = pc, 则pa*x + pb*y + pc*z = -1 --- 平面表示
        这里就是超定方程，最小二乘求解最优平面，求解参数是pa, pb ,pc --- normvec = (pa, pb, pc)T
        n = 根号下 pa*pa = pb*pb + pc*pc = 根号下 a*a/d*d + b*b/d*d + c*c/d*d = 根号下 (a*a +b*b +c*c)/d*d = 根号下 1/d*d = 1/d
        所以pca_result(3) = 1.0 / n = 1/(1/d) = d;
        整个函数就是求平面的4个参数：pa = a/d, pb = b/d, pc = c/d, d    ***  pca_result = (pa, pb, pc, d) ****
        物理意义： a, b, c是平面法向量的坐标分量值
    */
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    //求解超定方程：5个方程，3个未知数，这里求解最小二程，最优解//正常ax=b，ax-b=0，但是没有精准解，r=||ax-b||，r最小时的解
    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

float calc_dist(PointNomalType p1, PointNomalType p2)
{
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for(int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if(!Nearest_Points[i].empty() && flg_EKF_inited) 
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointNomalType downsample_result, mid_point;    //定义 降采样点，中点
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if(fabs(points_near[0].x - mid_point.x) > 0.5*filter_size_map_min && fabs(points_near[0].y-mid_point.y) > 0.5*filter_size_map_min && fabs(points_near[0].z-mid_point.z) > 0.5*filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for(int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if(points_near.size() < NUM_MATCH_POINTS) {break;}
                if(calc_dist(points_near[readd_i], mid_point) < dist) {need_add = false; break;}
            }
            if(need_add) {PointToAdd.push_back(feats_down_world->points[i]);}   
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

void publish_pre_imu(const StateGroup &state)
{
    Eigen::Quaterniond q = Eigen::Quaterniond(state.Rot);
    
    nav_msgs::Odometry imu_pre_odometry;
    imu_pre_odometry.header.frame_id = "base_link";
    imu_pre_odometry.child_frame_id = "/body";
    imu_pre_odometry.header.stamp = ros::Time().now();
    imu_pre_odometry.pose.pose.position.x = state.Pos(0);
    imu_pre_odometry.pose.pose.position.y = state.Pos(1);
    imu_pre_odometry.pose.pose.position.z = state.Pos(2);
    imu_pre_odometry.pose.pose.orientation.w = q.w();
    imu_pre_odometry.pose.pose.orientation.x = q.x();
    imu_pre_odometry.pose.pose.orientation.y = q.y();
    imu_pre_odometry.pose.pose.orientation.z = q.z();
    pub_pre_odometry.publish(imu_pre_odometry); //odometry是一个有方向的箭头（pose在header.frame_id坐标系下，twist再child_frame_id坐标系下）

    geometry_msgs::PoseStamped imu_pre_path;
    imu_pre_path.header.stamp = ros::Time().now();
    imu_pre_path.header.frame_id = "base_link";
    imu_pre_path.pose.position.x = state.Pos(0);
    imu_pre_path.pose.position.y = state.Pos(1);
    imu_pre_path.pose.position.z = state.Pos(2);
    imu_pre_path.pose.orientation.x = q.x();
    imu_pre_path.pose.orientation.y = q.y();
    imu_pre_path.pose.orientation.z = q.z();
    imu_pre_path.pose.orientation.w = q.w();
    IMU_path.header.frame_id = "base_link";
    
    //static int jjj = 0;
    //jjj++;
    //if(jjj % 10 == 0)
    //{
        IMU_path.poses.push_back(imu_pre_path);
        pub_pre_path.publish(IMU_path);             //path是一条连续的路径
    //}
}

void publish_pre_pointcloud(MeasureGroup &measurement)
{
            PointCloudXYZI pcl_pre = *(measurement.lidar);
            sensor_msgs::PointCloud2 pcl_pre_ros;
            pcl::toROSMsg(pcl_pre, pcl_pre_ros);
            pcl_pre_ros.header.frame_id = "base_link";
            pub_pre_pointcloud.publish(pcl_pre_ros);   
}
void publish_point_undis(PointCloudXYZI::Ptr pointcloud_undistort_waitpub)
{
            sensor_msgs::PointCloud2 pcl_ros;
            pcl::toROSMsg(*pointcloud_undistort_waitpub, pcl_ros);
            pcl_ros.header.frame_id = "base_link";  //每次发布都是同样的id，使得只能保持最新的
            pub_pointcloud.publish(pcl_ros);
}

void RGBpointBodyToWorld(PointNomalType const * const pi, PointNomalType * const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(g_state.Rot * (g_state.R_L_I*p_body + g_state.T_L_I) + g_state.Pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? pointcloud_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);   //每次的id都不同，使得能看到全局地图
    laserCloudmsg.header.frame_id = "base_link";
    pubLaserCloudFull.publish(laserCloudmsg);
}

//主函数
int main(int argc, char** argv)
{
    ros::init(argc,argv,"imuprocess");
    ros::NodeHandle nh;

    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());

    ros::Subscriber sub_imu = nh.subscribe( "/handsfree/imu",200000,imu_cbk);
    ros::Subscriber sub_pcl = nh.subscribe( "/velodyne_points", 200000, velodyne_cbk);

    pub_pre_odometry = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pub_pre_path = nh.advertise<nav_msgs::Path>("/Path_IMU", 100000);
    pub_pre_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/pre_pointcloud", 100000);
    pub_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/my_practice_pointcloud", 100000);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>       //发布正在扫描的点云topic
            ("/cloud_registered", 100000);

    G.setZero();
    H_T_H.setZero(); 
    I_STATE.setIdentity(); //变量初始化
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);//定义体素滤波器参数，体素边长为filter_size_surf_min = 0.5
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);   //提前设定好----0.5m

    ros::Rate rate(5000);       //5000hz = 0.02s = 20 ms
    bool status = ros::ok();
    while(status)
    {
        ros::spinOnce();
        if(buffer_to_meas(measurement))    //把imu信息从缓存器转到meas变量中 （注意！ 是地址传递，虽然动形参，但是等于动实参）
        {
            if(flg_first_scan)
            {
                first_lidar_time = measurement.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }
            publish_pre_pointcloud(measurement);
            p_imu->Process(measurement, g_state, pointcloud_undistort);
            publish_point_undis(pointcloud_undistort);
           //publish_pre_imu(g_state);

            if(0)
            {
                cout << "imu:" << endl;
                cout << "rot = \n" << g_state.Rot << "\npos = " << g_state.Pos.transpose() << "\nvel = " << g_state.Vel.transpose() << "\n ba:bg=" << g_state.acc_bias.transpose() << " " << g_state.gyr_bias.transpose() << endl;
                cout << "------" << endl;
            }

            // StateGroup state_propagate(g_state);
            // pos_lid = state_propagate.Pos + state_propagate.Rot * state_propagate.T_L_I;  //在世界坐标系下看雷达坐标原点——是三维向量(偏移向量*旋转 + 平移向量)（不管lidar和imu有没有rot，都是这个，因为是看原点——用向量考虑）
            
            // flg_EKF_inited = (measurement.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            // lasermap_fov_segment(); 
            // downSizeFilterSurf.setInputCloud(pointcloud_undistort);
            // downSizeFilterSurf.filter(*feats_down_body);                //体素滤波后的点云feats_down_body---是雷达坐标系下的点云
            // feats_down_size = feats_down_body->points.size();
            // initialize_map_kdtree(feats_down_size);                     //初始化ikdtree
            // if(initialized_map_kdtree_flag == true) {continue;}

            // int kdtree_size = ikdtree.size();
            // //int kdtree_size = ikdtree.validnum();
            // PointCloudXYZI::Ptr coeffSel_tmpt(new PointCloudXYZI(*feats_down_body));        
            // PointCloudXYZI::Ptr feats_down_update(new PointCloudXYZI(*feats_down_body));    
            // std::vector<double> res_last(feats_down_size, 1000.0);                            //初始化容器，feats_down_size个1000————用来装每个点的残差
            // if(kdtree_size >= 5)
            // {
            //     std::vector<bool>           point_selected_surf(feats_down_size, true);
            //     Nearest_Points.resize(feats_down_size);                //初始化而已
            //     feats_down_world->resize(feats_down_size);
            //     int rematch_num = 0;
            //     bool rematch_en = false;
            //     flg_EKF_converged = false;
            //     deltaR = 0.0; deltaT = 0.0;
            //     //double maximum_pt_range = 0.0;

            //     /*** ESIKF ***/
            //     for( iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++)         
            //     {
            //         laserCloudOri->clear();         //laserCloudOri是跟踪到了哪些点（最近邻成功找到，平面成功拟合，残差足够小）（雷达坐标系下）
            //         coeffSel->clear();
            //         /*** search the nearest point of each point and cal loss between them ***/
            //         for(int i = 0; i < feats_down_size; i++)                              
            //         {
            //             PointNomalType &point_body  = feats_down_body->points[i];   Eigen::Vector3d p_body(point_body.x, point_body.y, point_body.z);
            //             PointNomalType &point_world = feats_down_update->points[i];  //初始化--一会把点转到世界坐标下
            //             double body_pt_disTo_worldOri = sqrt(point_body.x*point_body.x + point_body.y*point_body.y + point_body.z*point_body.z);
            //             //maximum_pt_range = std::max(body_pt_disTo_worldOri, maximum_pt_range);
            //             pointBodyToWorld(&point_body, &point_world);//把点云从雷达坐标系转到世界坐标系

            //             std::vector<float> pointSearchSqDis(5);
            //             auto &points_near = Nearest_Points[i];      //points_near容器初始化 而以
            //             if(iterCount == 0 || rematch_en)
            //             {
            //                 //point_selected_surf[i] = true;
            //                 ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);   //基准点,找几个点,这几个点,这几个点的距离   //最近点放到points_near
            //                 //float max_distance = pointSearchSqDis[NUM_MATCH_POINTS - 1];
            //                 //if(max_distance > 1) {point_selected_surf[i] = false;}
            //                 point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS        ? false
            //                                          : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 3 ? false
            //                                                                                       : true;
            //             }
            //             if(point_selected_surf[i] == false){continue;}

            //             Eigen::Matrix<float, 4, 1> pabcd;
            //             point_selected_surf[i] = false;
            //             if(esti_plane(pabcd, points_near, 0.2f))
            //             {
            //                 float pd2 = pabcd(0)*point_world.x + pabcd(1)*point_world.y + pabcd(2)*point_world.z + pabcd(3);//基准点到平面距离（没推出来）
            //                 //cout << "body_pt_disTo_worldOri = " << body_pt_disTo_worldOri << endl;
            //                 float s = 1 - 0.9*fabs(pd2) / sqrt(p_body.norm());
            //                 if(s > 0.9)
            //                 {
            //                     point_selected_surf[i] = true;
            //                     coeffSel_tmpt->points[i].x = pabcd(0);
            //                     coeffSel_tmpt->points[i].y = pabcd(1);
            //                     coeffSel_tmpt->points[i].z = pabcd(2);
            //                     coeffSel_tmpt->points[i].intensity = pd2;
            //                     res_last[i] = abs(pd2);
            //                 }

            //                 // double acc_distance = body_pt_disTo_worldOri < 500 ? m_maximum_res_dis : 1.0;
            //                 // if(pd2 < acc_distance)
            //                 // {
            //                 //     point_selected_surf[i] = true;
            //                 //     coeffSel_tmpt->points[i].x = pabcd(0);
            //                 //     coeffSel_tmpt->points[i].y = pabcd(1);
            //                 //     coeffSel_tmpt->points[i].z = pabcd(2);
            //                 //     coeffSel_tmpt->points[i].intensity = pd2;
            //                 //     res_last[i] = abs(pd2);
            //                 // }
            //             }
            //         }
            //         double total_residual = 0.0;
            //         effct_feat_num = 0; 
            //         for(int i = 0; i < coeffSel_tmpt->points.size(); i++)
            //         {   
            //             if(point_selected_surf[i] && (res_last[i] <= 2.0))
            //             {
            //                 laserCloudOri->push_back(feats_down_body->points[i]);   //哪些点能成功被计算（最近邻成功找到，平面成功拟合，残差足够小）
            //                 coeffSel->push_back(coeffSel_tmpt->points[i]);          //离他最近的平面是什么样的，追加进来
            //                 total_residual = total_residual + res_last[i];
            //                 effct_feat_num++;
            //             }
            //         }
            //         /*** computation of Measurement Jacobian matrix H and Measurements vector ***/
            //         Eigen::MatrixXd Hsub(effct_feat_num, 6);    //根据有效残差点数量构建大矩阵(effct_feat_num行， 6列)
            //         Eigen::VectorXd meas_vec(effct_feat_num);   //观测向量set_initial_state_cov( StatesGroup &state )
            //         Hsub.setZero();
            //         for(int i = 0; i < effct_feat_num; i++)
            //         {
            //             const PointNomalType &laserCloudOri_Onepoint = laserCloudOri->points[i];   //观测原始点云中的一个点（雷达坐标系下）
            //             Eigen::Vector3d point_this_be(laserCloudOri_Onepoint.x, laserCloudOri_Onepoint.y, laserCloudOri_Onepoint.z);
            //             Eigen::Vector3d point_this = g_state.R_L_I * point_this_be + g_state.T_L_I;                             //坐标转换：雷达->局部imu    //因为avia 雷达和imu只有平移
            //             Eigen::Matrix3d point_this_matrix;
            //             point_this_matrix << SKEW_SYM_MATRIX(point_this);          //反对称矩阵---imu坐标系下看到的基准点

            //             /*** get the normal vector of closest surface/corner ***/
            //             const PointNomalType &norm_p = coeffSel->points[i];                         //平面参数（包括残差）
            //             Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);                     //---原地图点拟合的平面参数

            //             /*** calculate the Measuremnt Jacobian matrix H ***/
            //             Eigen::Vector3d C(g_state.Rot.transpose() * norm_vec);
            //             Eigen::Vector3d A(point_this_matrix * C);  //基准点的反对称
            //             Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;             //Hsub第一行是A(3个数)，平面参数pa，pb，pc

            //             /*** Measuremnt: distance to the closest surface/corner ***/
            //             meas_vec(i) = -norm_p.intensity;                                             //点到平面残差---ESIKF的观测向量-  -z
            //         }
            //         Eigen::Vector3d rot_add, t_add;
            //         Eigen::Matrix<double, DIM_STATE, 1> solution;
            //         Eigen::MatrixXd K(DIM_STATE, effct_feat_num);

            //         /*** Iterative Kalman Filter Update ***/ 
            //         if(flg_EKF_inited == false)
            //         {
            //             cout << "ESIKF update init..." << endl;
            //             set_initial_state_cov(g_state);
            //         }
            //         else
            //         {
            //             auto &&Hsub_T = Hsub.transpose();           //Hsub点数行，6列  //Hsub_T 6行，x列
            //             H_T_H.block<6,6>(0,0) = Hsub_T * Hsub;      // 6*x * x*6 = 6*6
            //             Eigen::Matrix<double, DIM_STATE, DIM_STATE> &&K_1 = (H_T_H + (g_state.cov/LASER_POINT_COV).inverse()).inverse();    //论文中(14)的括号里面的R为单位矩阵
            //             K = K_1.block<DIM_STATE, 6>(0,0) * Hsub_T;  //K = (K_1)*H^t //18*6 * 6行 点数个列 = 18*x

            //             auto vec = state_propagate - g_state;       //误差状态 = 名义状态 - 真实状态    //第一次迭代 state_propagate = g_state，然后更新了g_state，vec就不是0了
            //             //auto vec = g_state - state_propagate;
            //             //solution = K * (meas_vec - Hsub * vec.block<6,1>(0,0)) + vec;     //detal x = K(-z - h(x)) // 公式14
            //             solution = K * meas_vec - K * Hsub * vec.block<6, 1>(0, 0) + vec;
            //             g_state = g_state + solution;       //真实状态 = 名义状态 + 误差状态
            //             rot_add = solution.block<3,1>(0,0);         //用来判断是否收敛---增加的旋转 位移 已经不多了，没什么变化---则视为收敛
            //             t_add = solution.block<3,1>(3,0);   
            //             flg_EKF_converged = false;
            //             if( (rot_add.norm()*57.3)<0.01 && (t_add.norm()*100 <0.015) )  //需要调试
            //             {
            //                 flg_EKF_converged = true;
            //             }
            //             //deltaR = rot_add.norm()*57.3;
            //             //deltaT = t_add.norm()*100;
            //         }

            //         /*** Rematch Judgement ***/
            //         //g_state.last_update_time = measurement.lidar_end_time;
            //         rematch_en = false;
            //         if(flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS-2))))
            //         {
            //             rematch_en = true;
            //             rematch_num++;
            //         }

            //         /*** Convergence Judgements and Covariance Update ***/
            //         if(rematch_num >= 2 || (iterCount == (NUM_MAX_ITERATIONS -1)))
            //         {
            //             if(flg_EKF_inited == true)
            //             {
            //                 /*** Covariance Update ***/ 
            //                 G.block<DIM_STATE, 6>(0,0) = K*Hsub;        //18行，点数列 * 点数行，6列 =18*6 
            //                 g_state.cov = (I_STATE - G) * g_state.cov;  //公式15：P = (I-KH) * P    //18*18 = 18*18*18*18

            //             }
            //             break;
            //         }
            //     }
            // }
            // if(0)
            // {
            //     cout << "esikf:" << endl;
            //     cout << "rot = \n" << g_state.Rot << "\npos = " << g_state.Pos << "\nvel = " << g_state.Vel << "\n ba:bg=" << g_state.acc_bias << " " << g_state.gyr_bias << endl;
            // }
            // /*** add new frame points to map ikdtree ***/
            // map_incremental();
            publish_pre_imu(g_state);
            publish_frame_world(pubLaserCloudFull);
        }
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}