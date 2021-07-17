#include <sobfu/sob_fusion2.hpp>

// #include <opencv2/opencv.hpp>

SobFusion2::SobFusion2(const Params &params) : frame_counter_(0), params(params) {
    int cols = params.cols;
    int rows = params.rows;

    // dists_.create(rows, cols);
    dists_vec_.resize(params.n_cams);
    curr_.depths.resize(params.n_cams);
    // curr_.normals_pyr.resize(1);

    // prev_.depth_pyr.resize(1);
    // prev_.normals_pyr.resize(1);

    // curr_.points_pyr.resize(1);
    // prev_.points_pyr.resize(1);

    for (int i = 0; i < params.n_cams; i++)
    // for (int i = 0; i < 4; i++)
    {
        dists_vec_[i].create(rows, cols);
        curr_.depths[i].create(rows, cols);
    }
    // curr_.depth_pyr[0].create(rows, cols);
    // curr_.normals_pyr[0].create(rows, cols);

    // prev_.depth_pyr[0].create(rows, cols);
    // prev_.normals_pyr[0].create(rows, cols);

    // curr_.points_pyr[0].create(rows, cols);
    // prev_.points_pyr[0].create(rows, cols);

    // depths_.create(rows, cols);
    // normals_.create(rows, cols);
    // points_.create(rows, cols);

    // poses_.clear();
    // poses_.reserve(4096);
    // poses_.push_back(cv::Affine3f::Identity());

    mc = cv::Ptr<kfusion::cuda::MarchingCubes>(new kfusion::cuda::MarchingCubes());
    mc->setPose(params.volume_pose);
}

SobFusion2::~SobFusion2() = default;

Params &SobFusion2::getParams() { return params; }

pcl::PolygonMesh::Ptr SobFusion2::get_phi_global_mesh() { return get_mesh(phi_global); }

pcl::PolygonMesh::Ptr SobFusion2::get_phi_global_psi_inv_mesh() { return get_mesh(phi_global_psi_inv); }

pcl::PolygonMesh::Ptr SobFusion2::get_phi_n_mesh() { return get_mesh(phi_n); }

pcl::PolygonMesh::Ptr SobFusion2::get_phi_n_psi_mesh() { return get_mesh(phi_n_psi); }

std::shared_ptr<sobfu::cuda::DeformationField> SobFusion2::getDeformationField() { return this->psi; }

/* PIPELINE
 *
 * --- frame 0 ---
 *
 * 1. bilateral filter
 * 2. depth truncation
 * 3. initailisation of phi_global and phi_n
 *
 * --- frames n + 1 ---
 * 1. bilateral filter
 * 2. depth truncation
 * 3. initialisation of phi_n
 * 4. estimation of psi
 * 5. fusion of phi_n(psi)
 * 6. warp of phi_global with psi^-1
 *
 */

// bool SobFusion::operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image & /*image*/) {
bool SobFusion2::operator()(const std::vector<kfusion::cuda::Depth> &depths) {
    std::cout << "--- FRAME NO. " << frame_counter_ << " ---" << std::endl;

    for (int i = 0; i < params.n_cams; i++)
    {
        /*
         *  bilateral filter 双边滤波去噪
         */

        kfusion::cuda::depthBilateralFilter(depths[i], curr_.depths[i], params.bilateral_kernel_size,
                                            params.bilateral_sigma_spatial, params.bilateral_sigma_depth);

        /*
         * depth truncation 截断深度
         */

        kfusion::cuda::depthTruncation(curr_.depths[i], params.icp_truncate_depth_dist);

        /*
         *  compute distances using depth map 计算距离，不太理解，类似于计算点云坐标
         */

        kfusion::cuda::computeDists(curr_.depths[i], dists_vec_[i], params.intrs[i]);
        
        /* visualization */
        // cv::Mat display_t(720,1280,CV_32F);
        // dists_vec_[i].download(display_t.data, display_t.step);
        // display_t.convertTo(display_t, CV_8U, 255.0/3);
        // cv::imshow("dists",display_t);
        // cv::waitKey();
    }

    if (frame_counter_ == 0) {
        /*
         * INITIALISATION OF PHI_GLOBAL 初始化参考体积
         */

        phi_global = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        for (int i = 0; i < params.n_cams; i++)
        {
            phi_global->integrate(dists_vec_[i], params.cam_poses[i], params.intrs[i]);
            // std::cout<<params.cam_poses[i].matrix<<std::endl;
            // std::cout<<"([f = " << params.intrs[i].fx << ", " << params.intrs[i].fy << "] [cp = " << params.intrs[i].cx << ", " << params.intrs[i].cy << "])"<<std::endl;
        }
        

        /*
         * INITIALISATION OF PHI_GLOBAL(PSI_INV), PHI_N, AND PHI_N(PSI)
         * 分别为  从参考帧融合当前数据之后再变形到当前帧，用于输出显示
         *        当前数据帧
         *        变形后的数据帧（和参考帧对齐）
         */

        phi_global_psi_inv = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n              = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n_psi          = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        /*
         * INITIALISATION OF PSI AND PSI_INV 初始化向前和向后的两个变形场
         */

        this->psi     = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);
        this->psi_inv = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);

        /*
         * INITIALISATION OF THE SOLVER 初始化求解器
         */

        this->solver = std::make_shared<sobfu::cuda::Solver>(params);

        return ++frame_counter_, true;
    }

    /*
     * UPDATE OF PHI_N PHI_N为数据帧或叫live帧，即当前帧
     */

    phi_n->clear();
    for (int i = 0; i < params.n_cams; i++)
        phi_n->integrate(dists_vec_[i], params.cam_poses[i], params.intrs[i]);

    /*
     * ESTIMATION OF DEFORMATION FIELD AND SURFACE FUSION
     * 估计变形场，返回的是变形后的当前帧，与参考帧对齐。然后融合入参考体积
     */

    if (frame_counter_ < params.start_frame) {
        this->phi_global->integrate(*phi_n);
        return ++frame_counter_, true;
    }

    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);
    this->phi_global->integrate(*phi_n_psi);

    return ++frame_counter_, true;
}

pcl::PolygonMesh::Ptr SobFusion2::get_mesh(cv::Ptr<kfusion::cuda::TsdfVolume> vol) {
    kfusion::device::DeviceArray<pcl::PointXYZ> vertices_buffer_device;
    kfusion::device::DeviceArray<pcl::Normal> normals_buffer_device;

    /* run marching cubes */
    std::shared_ptr<kfusion::cuda::Surface> model =
        std::make_shared<kfusion::cuda::Surface>(mc->run(*vol, vertices_buffer_device, normals_buffer_device));
    kfusion::cuda::waitAllDefaultStream();

    pcl::PolygonMesh::Ptr mesh = convert_to_mesh(model->vertices);
    return mesh;
}

pcl::PolygonMesh::Ptr SobFusion2::convert_to_mesh(const kfusion::cuda::DeviceArray<pcl::PointXYZ> &triangles) {
    if (triangles.empty()) {
        return pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width  = static_cast<int>(triangles.size());
    cloud.height = 1;
    triangles.download(cloud.points);

    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
    pcl::toPCLPointCloud2(cloud, mesh->cloud);

    mesh->polygons.resize(triangles.size() / 3);
    for (size_t i = 0; i < mesh->polygons.size(); ++i) {
        pcl::Vertices v;
        v.vertices.push_back(i * 3 + 0);
        v.vertices.push_back(i * 3 + 1);
        v.vertices.push_back(i * 3 + 2);
        mesh->polygons[i] = v;
    }

    return mesh;
}
