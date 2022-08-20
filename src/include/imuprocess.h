#pragma once    //保证头文件只被编译一次
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <Eigen/Geometry>

#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "comman_lib.h"
#include "so3_math.h"

#define COV_OMEGA_NOISE_DIAG 1e-1
#define COV_ACC_NOISE_DIAG 0.4
#define COV_GYRO_NOISE_DIAG 0.2

#define COV_BIAS_ACC_NOISE_DIAG 0.05
#define COV_BIAS_GYRO_NOISE_DIAG 0.1

using namespace std;

const bool timesize(PointNomalType &x, PointNomalType &y)   //用于sort的纺函数
{
    return (x.curvature < y.curvature);
};

class ImuProcess
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     //保证内存对齐（march -native就是内存不对齐）//申请内存时重写operator new
    ImuProcess();
    ~ImuProcess();

    void reset();
    void reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
    void set_gyr_cov();
    void set_acc_cov();
    void set_gyr_bias_cov();
    void set_acc_bias_cov();
    void set_extrinsic(const Eigen::Vector3d &transl, const Eigen::Matrix3d &rot);  
    void set_extrinsic(const Eigen::Vector3d &transl);
    void set_extrinsic(const Eigen::Matrix<double,4,4> &T);
    Eigen::Matrix<double, 12, 12> Q;
    void Process(const MeasureGroup &meas, StateGroup &state, PointCloudXYZI::Ptr &pointcloud_undistort);             //imu处理的主要函数

    Eigen::Matrix<double, 3, 1> cov_acc;    //协方差的维数 和 随机变量的个数有关 （这里不是协方差矩阵，是协方差。多个这样的在一起才是协方差矩阵，其每个元素是向量之间的协方差）
    Eigen::Matrix<double, 3, 1> cov_gyr;    //陀螺仪协方差
    Eigen::Matrix<double, 3, 1> cov_bias_acc;
    Eigen::Matrix<double, 3, 1> cov_bias_gyr;
    Eigen::Matrix<double, 3, 1> mean_acc;   //平均加速度，用于中值离散积分
    Eigen::Matrix<double, 3, 1> mean_gyr;   
    Eigen::Matrix<double, 3, 1> angvel_last;
    Eigen::Matrix<double, 3, 1> acc_last;
    Eigen::Matrix<double, 3, 3> Lidar_R_wrt_IMU;        //外参旋转
    Eigen::Vector3d Lidar_T_wrt_IMU;        //外参平移
    double start_timestamp;
    int init_process_nums = 1;
    bool is_first_frame = true;
    bool imu_need_init = true;
    double last_lidar_end_time_;//20ms中雷达的最后一次采样时间
    double first_lidar_time;  //雷达第一次的采样时间


    void IMU_init(const MeasureGroup &meas, StateGroup &state, int &N);                                             //初始化imu函数
    StateGroup imu_preintegration(const StateGroup &state_in, std::deque< sensor_msgs::Imu::ConstPtr > &v_imu, double end_pose_dt, const double pcl_beg_time); //预积分函数（前向传播里的）//函数声明
    void pointcloud_undistort_func(StateGroup &state, PointCloudXYZI &pcl_out);
    void IMU_propagation_and_undistort(const MeasureGroup &meas, StateGroup &state, PointCloudXYZI &pcl_out);                                              //前向传播函数
    sensor_msgs::ImuConstPtr last_imu_;     //用于更新指针所指  //ImuConstPtr是智能指针
    deque<sensor_msgs::ImuConstPtr> v_imu_;
    vector<Pose6D> IMUpose;                 //仅用于往回迭代做点云去畸变
};

//构造函数
ImuProcess::ImuProcess() :  is_first_frame(true), imu_need_init(true), start_timestamp(-1) 
{
    init_process_nums = 1;
    Q = Eigen::Matrix<double, 12, 12>::Zero();
    cov_acc = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1); //赋初始值
    cov_gyr = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1);
    cov_bias_acc = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1);
    cov_bias_gyr = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1);
    mean_acc = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1);
    mean_gyr = Eigen::Matrix<double, 3, 1>(0.1, 0.1, 0.1);
    angvel_last = Eigen::Matrix<double, 3, 1>::Zero();
    Lidar_R_wrt_IMU = Eigen::Matrix3d::Identity();
    Lidar_T_wrt_IMU = Eigen::Vector3d::Zero();
    last_imu_.reset(new sensor_msgs::Imu());       //当调用reset（new xxx())重新赋值时，智能指针指向新的对象
}
//析构
ImuProcess::~ImuProcess() {}

//reset imu函数
void ImuProcess::reset()
{
    cout << " reset imu " << endl;
    mean_acc = Eigen::Matrix<double, 3, 1>(0, 0, -1);
    mean_gyr = Eigen::Matrix<double, 3, 1>(0, 0, 0);
    cov_acc = Eigen::Vector3d(0.1, 0.1, 0.1);
    cov_gyr = Eigen::Vector3d(0.1, 0.1, 0.1);
    angvel_last = Eigen::Matrix<double, 3, 1>::Zero();
    start_timestamp = -1;
    init_process_nums = 1;
    IMUpose.clear();
    is_first_frame = true;
    last_imu_.reset(new sensor_msgs::Imu());
    imu_need_init = true;                       //重置时 需重新初始化
    v_imu_.clear();
}

//避免异常值（vel）输入函数
bool check_state_vel(StateGroup &state)
{
    bool is_fail = false;
    for (int i = 0; i < 3; i++)
    {
        if(fabs(state.Vel(i)) > 10)
        {
            is_fail = true;
            state.Vel(i) = 0.0;
            //cout << "check the outlier: " << "state.vel(" << i << ") = "  << state.Vel(i) << endl;
        }
    }
    return is_fail;
}
//避免异常值（pos）输入函数
void check_state_pos(const StateGroup &state_in, StateGroup &state_out)
{
    if( (state_in.Pos - state_out.Pos).norm() > 1.0 )
    {
        state_out.Pos = state_in.Pos;       //别写反了，写反了就是 in在前面，但是in是const，没法改变
        //cout << "check the outlier: " << "state_in.pos = "  << state_in.Pos.transpose() << ";  state_in.pos = " << state_out.Pos.transpose() << endl;
    }
}

void ImuProcess::set_extrinsic(const Eigen::Matrix<double,4,4> &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}
void ImuProcess::set_extrinsic(const Eigen::Vector3d &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}
void ImuProcess::set_extrinsic(const Eigen::Vector3d &transl, const Eigen::Matrix3d &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

//imu初始化函数
void ImuProcess::IMU_init(const MeasureGroup &meas, StateGroup &state, int &N)
{
    Eigen::Matrix<double, 3, 1> current_acc, current_gyr;
    if(is_first_frame)
    {
        reset();
        N = 1;
        is_first_frame = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &imu_gyr = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;
        first_lidar_time = meas.lidar_beg_time;
    }
    for (const auto &i : meas.imu)  //增强for循环 ———— 遍历deque容器
    {
        const auto &imu_acc = i->linear_acceleration;
        const auto &imu_gyr = i->angular_velocity;
        current_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //函数形参用了cosnt，导到新变量里做
        current_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;

        mean_acc = mean_acc + (current_acc - mean_acc)/N;
        mean_gyr = mean_gyr + (current_gyr - mean_gyr)/N;
        cov_acc = cov_acc*(N-1.0)/N + (current_acc - mean_acc).cwiseProduct(current_acc-mean_acc) * (N-1.0)/(N*N);
        cov_gyr = cov_gyr*(N-1.0)/N + (current_gyr - mean_gyr).cwiseProduct(current_gyr-mean_gyr) * (N-1.0)/(N*N);
        N++;
    }
    state.gyr_bias = mean_gyr;                      //陀螺仪偏置
    state.gravity = -mean_acc / mean_acc.norm() * 9.805;   //初始重力
    state.Rot = Eigen::Matrix3d::Identity();        //初始旋转
    state.R_L_I = Lidar_R_wrt_IMU;
    state.T_L_I = Lidar_T_wrt_IMU;
    last_imu_ = meas.imu.back();     
}

std::mutex g_imu_premutex;
StateGroup ImuProcess::imu_preintegration(const StateGroup &state_in, std::deque< sensor_msgs::Imu::ConstPtr > &v_imu, double end_pose_dt, const double pcl_beg_time)
{
    //std::unique_lock< std::mutex > lock( g_imu_premutex );  //没必要上锁应该是

    StateGroup state_out = state_in;
    if(check_state_vel(state_out))          //check the outlier
    {
        //state_out.display(state_out, "state_out");
        //state_in.display(state_in, "state_in");
    }

    Eigen::Matrix<double,3, 1> acc_imu_, angvel_avr_, acc_avr_, vel_imu_, pos_imu_;  //创建一些中间变量 中间计算用
    Eigen::Matrix3d rot_imu_;   double dt = 0;      int if_first_imu_ = 1;
    Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity());
    Eigen::MatrixXd cov_w(Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero());     //tocheck****
    vel_imu_ = state_out.Vel;
    pos_imu_ = state_out.Pos;
    rot_imu_ = state_out.Rot;

    for (std::deque<sensor_msgs::Imu::ConstPtr>::iterator it_imu = v_imu.begin(); it_imu != ( v_imu.end()-1 ); it_imu++)
    {
        sensor_msgs::Imu::ConstPtr head = *( it_imu );
        sensor_msgs::Imu::ConstPtr tail = *( it_imu + 1 );
        if(tail->header.stamp.toSec() < last_lidar_end_time_) {continue;}

        //计算ak，wk （中值）
        angvel_avr_ << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y), 
                       0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr_    << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                       0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
        angvel_avr_ = angvel_avr_ - state_out.gyr_bias;          //平均角速度 - 陀螺偏置
        acc_avr_    = acc_avr_ * 9.805/mean_acc.norm()  - state_out.acc_bias;          //平均加速度 - 加速度偏置（加速度偏置忽略 = 0）
        //acc_avr_    = acc_avr_   - state_out.acc_bias; 
        //计算dt
        if(head->header.stamp.toSec() < last_lidar_end_time_)
        {
            //if_first_imu_ = 0;  //清除标志位
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;   //自己写遇到的问题1：忘记对last_update_time进行更新，或许是抖动或不准的原因（每组第一帧的dt都是和0的时间）
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }
        if( dt > 0.05 ) //防止非线性误差过大
        {
            dt = 0.05;
        }
        /* covariance propagation */ 
        Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d acc_avr_skew;
        acc_avr_skew << SKEW_SYM_MATRIX( acc_avr_ );
        Eigen::Matrix3d Exp_f = Exp( angvel_avr_, -dt );  //角速度平均值 只用到了这一个地方
        F_x.block<3,3>(0,0) = Exp_f.transpose();
        F_x.block<3,3>(0,9) = -Jr_omega_dt * dt;
        F_x.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
        F_x.block<3,3>(3,6) = Eigen::Matrix3d::Identity() * dt;
        F_x.block<3,3>(6,0) = -rot_imu_ * acc_avr_skew * dt;
        F_x.block<3,3>(6,12) = -rot_imu_ * dt;
        F_x.block<3,3>(6,15) = Eigen::Matrix3d::Identity() * dt;

        Eigen::Matrix3d cov_omega_diag, cov_acc_diag, cov_gyr_diag;
        cov_omega_diag = Eigen::Vector3d( COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG ).asDiagonal();
        cov_acc_diag = Eigen::Vector3d( COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG ).asDiagonal();
        cov_gyr_diag = Eigen::Vector3d( COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG ).asDiagonal();
        //cov_acc_diag.setIdentity();cov_gyr_diag.setIdentity();
        //cov_acc_diag.diagonal() = cov_acc;
        //cov_gyr_diag.diagonal() = cov_gyr;
        cov_w.block< 3, 3 >( 0, 0 ) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;
        cov_w.block< 3, 3 >( 3, 3 ) = rot_imu_ * cov_gyr_diag * rot_imu_.transpose() * dt * dt;
        cov_w.block< 3, 3 >( 6, 6 ) = cov_acc_diag * dt * dt;
        cov_w.block< 3, 3 >( 9, 9 ).diagonal() = Eigen::Vector3d(COV_BIAS_GYRO_NOISE_DIAG,COV_BIAS_GYRO_NOISE_DIAG,COV_BIAS_GYRO_NOISE_DIAG)*dt*dt;
        cov_w.block< 3, 3 >(12,12).diagonal()  = Eigen::Vector3d(COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG )*dt*dt;

        state_out.cov = F_x * state_out.cov * F_x.transpose() + cov_w;

        //计算P V Q
        rot_imu_ = rot_imu_ * Exp_f;
        acc_imu_ = rot_imu_ * acc_avr_ + state_out.gravity;
        pos_imu_ = pos_imu_ + vel_imu_ * dt + 0.5 * acc_imu_ * dt * dt;
        vel_imu_ = vel_imu_ + acc_imu_ * dt;
        
        //利用完毕————更新数据————把当前帧变为上一帧
        angvel_last = angvel_avr_;  //仅用于点云校正
        acc_last    = acc_imu_ ;    

        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    //此帧imu 离 此20ms雷达开始时间 = 偏移时间，反向传播时拿出来用
        IMUpose.push_back(set_pose6d(offs_t, acc_imu_, angvel_avr_, vel_imu_, pos_imu_, rot_imu_));
    }
    dt = end_pose_dt;
    //state_out.last_update_time = v_imu.back()->header.stamp.toSec();    //更新上一组最后一帧imu的时间
    if(dt > 0.1)
    {
        dt = 0.1;
    }
    state_out.Vel = vel_imu_ + acc_imu_ * dt;
    state_out.Rot = rot_imu_ * Exp(angvel_avr_, dt);
    state_out.Pos = pos_imu_ + vel_imu_ * dt + 0.5 * acc_imu_ * dt * dt;

    if(check_state_vel(state_out))
    {
        //state_out.display(state_out, "state_out");
        //state_in.display(state_in, "state_in");     //state_in 是const对象，const对象只能调用const函数 void 函数() const {} 要么！就加入static关键字————变成静态成员函数，
                                                    //静态成员函数 为类的全部而服务，不为某一个特定的对象服务（也就是说我不管你这对象是不是const，我都好使）
    }
    check_state_pos(state_in, state_out);
    return state_out;
}

void ImuProcess::pointcloud_undistort_func(StateGroup &in_state, PointCloudXYZI &pcl_out)
{
    auto it_pcl = pcl_out.points.end() - 1; //按时间排好序了
    for(auto it_imu_pose = IMUpose.end()-1; it_imu_pose != IMUpose.begin(); it_imu_pose--)
    {
        Eigen::Matrix<double,3, 1> acc_imu_( 0, 0, 0 ), angvel_avr_( 0, 0, 0 ), acc_avr_( 0, 0, 0 ), vel_imu_( 0, 0, 0 ), pos_imu_( 0, 0, 0 );  //创建一些中间变量 中间计算用
        Eigen::Matrix3d rot_imu_;

        auto head = it_imu_pose -1;
        auto tail = it_imu_pose;
        rot_imu_ << head->rot[0], head->rot[1], head->rot[2], head->rot[3], head->rot[4], head->rot[5], head->rot[6], head->rot[7], head->rot[8];
        acc_imu_ << tail->acc[0], tail->acc[1], tail->acc[2];
        vel_imu_ << head->vel[0], head->vel[1], head->vel[2];
        pos_imu_ << head->pos[0], head->pos[1], head->pos[2];
        angvel_avr_ << tail->gyr[0], tail->gyr[1], tail->gyr[2];

        for ( ; it_pcl->curvature/double(1000) > head->offset_time; it_pcl--)   //按时间倒叙处理每一个点    curvature点的偏移时间，offset_time是imu帧的偏移时间
        {
            double dt = it_pcl->curvature/double(1000) - head->offset_time;
            Eigen::Vector3d p_i(it_pcl->x,it_pcl->y, it_pcl->z);
            Eigen::Matrix3d R_i(rot_imu_ * Exp(angvel_avr_, dt));   //这个点处的i时刻IMU旋转
            Eigen::Vector3d T_ei( pos_imu_ + vel_imu_*dt + 0.5*acc_imu_*dt*dt - in_state.Pos );
            Eigen::Vector3d p_compensate = in_state.R_L_I.transpose() * (in_state.Rot.transpose() * (R_i*(in_state.R_L_I*p_i+in_state.T_L_I) + T_ei) - in_state.T_L_I);

            it_pcl->x = p_compensate(0);
            it_pcl->y = p_compensate(1);
            it_pcl->z = p_compensate(2);
            if(it_pcl == pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::IMU_propagation_and_undistort(const MeasureGroup &meas, StateGroup &state, PointCloudXYZI &pcl_out)
{
    //上一组的最后帧 给到 当前组的第一帧 （测量量）
    auto v_imu = meas.imu;          // 不能改const里面的，要换一个碗
    v_imu.push_front(last_imu_);    // 把上一组的imu尾帧测量 给到 当前组的第一帧（改变了这组imu的第一帧，但是后续的所有测量都没有变）
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //20ms中的末尾imu时间
    const double &pcl_beg_time = meas.lidar_beg_time;                   //点云的第一个时间（从测量值中读取出来的）
    const double &pcl_end_time = meas.lidar_end_time;                   //点云的最后一个时间
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    double end_pose_dt = note * (pcl_end_time - imu_end_time);
    last_lidar_end_time_ = pcl_end_time;

    pcl_out = *(meas.lidar);
    sort(pcl_out.points.begin(), pcl_out.points.end(), timesize); //纺函数不加（）
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_last, angvel_last, state.Vel, state.Pos, state.Rot)); //初始化IMUpose

    state = imu_preintegration(state, v_imu, end_pose_dt, pcl_beg_time);       //imu前向传播
    pointcloud_undistort_func(state, pcl_out);      //点云反向传播
    
    //cout << "---meas: " << meas.lidar->front().x << "/" << meas.lidar->front().y << "/" << meas.lidar->front().z << endl;         //这里可以用python做数据可视化分析
    //cout <<"---out: " << pcl_out.points.front().x << "/" << pcl_out.points.front().y << "/" << pcl_out.points.front().z << endl;

    last_imu_ = meas.imu.back();
    //cout << "state is : " << "\nstate.Pos = " << state.Pos.transpose() << "\nstate.Vel = " << state.Vel.transpose() << "\nstate.Rot = \n" << state.Rot << endl;
}

//处理imu的主函数----------------------------------------------------------
void ImuProcess::Process(const MeasureGroup &meas, StateGroup &state, PointCloudXYZI::Ptr &pointcloud_undistort)  //MeasureGroup如果在main中，就会出错————头文件重复包含（这里需要main，而main又包含imuprocess，所以单独做一个工具类）
{
    if(meas.imu.empty())
    {
        cout << "meas.imu = empty" << endl;
        return;
    }
    if(imu_need_init)
    {
        IMU_init(meas, state, init_process_nums);
        
        last_imu_ = meas.imu.back();
        imu_need_init = true;
        if(init_process_nums > 10)
        {
            imu_need_init = false;
            cout << "IMU init successed: \n" << "Gravity: " << state.gravity.transpose() << "\n" << "gyr_bias: " << state.gyr_bias.transpose() << endl;
        }
        return;
    }
    IMU_propagation_and_undistort(meas, state, *pointcloud_undistort);    //pointcloud_undistort是指针，解引用，那边用引用接收，接收这个是新的变量的地址了吧？
    last_imu_ = meas.imu.back();
}