#pragma warning(disable: 4819)

#include <seeta/FaceEngine.h>
#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>
#include <chrono>

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <dirent.h>

using namespace std;

int parse_args(int argc,char *argv[],float& thresh,int minface,string& targets,string& peoples)
{
    if(argc < 3) {
	std::cerr << "usage: " << argv[0] << " <target dir> <people dir> [thresh:0.7(def)] [minface:40(def)]" << std::endl;
	::exit(-1);
    }
    targets = argv[1];
    peoples = argv[2];

    if(targets.empty() || peoples.empty()) {
	std::cerr << "usage: " << argv[0] << " <target dir> <people dir> [thresh:0.7(def)] [minface:40(def)]" << std::endl;
	::exit(-1);
    }
    if(argc >= 4)
	thresh = atof(argv[3]);
    if(argc >= 5)
	minface = atof(argv[4]);
    std::cout << "Threshhold = " << thresh << ", MinFaceSize = " << minface << std::endl;
    return 0;
}
int parse_image_list(string dir,std::vector<string>& list)
{
    DIR *_dir;
    struct  dirent *ptr;
    _dir = opendir(dir.c_str()); ///open the dir
    if(_dir == nullptr) {
	std::cerr << "Open Directory " << dir << " fail! " << std::endl;
	::exit(-1);
    }
    while((ptr = readdir(_dir)) != nullptr) ///read the list of this dir
    {
	//printf("d_type:%d d_name: %s\n", ptr->d_type,ptr->d_name);
	if(ptr->d_type == DT_REG && (strstr(ptr->d_name,".jpg") || strstr(ptr->d_name,".png")))
	{
	    list.emplace_back(dir + "/" + ptr->d_name);
	}
    }
    closedir(_dir);
    return 0;
}

int main(int argc,char *argv[])
{
    // recognization threshold
    float threshold = 0.7f;
    int minface = 40;
    string target_dir,people_dir;

    parse_args(argc,argv,threshold,minface,target_dir,people_dir);

    int id = 0;
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    seeta::ModelSetting FD_model( "./model/fd_2_00.dat", device, id );
    seeta::ModelSetting PD_model( "./model/pd_2_00_pts5.dat", device, id );
    seeta::ModelSetting FR_model( "./model/fr_2_10.dat", device, id );
    seeta::FaceEngine engine( FD_model, PD_model, FR_model, 2, 16 );

    //set face detector's min face size
    engine.FD.set( seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, minface);

    std::cout << "----register target directory: "<< target_dir << std::endl;
    std::vector<std::string> GalleryImageFilename ;//= { "1.jpg" };
    parse_image_list(target_dir,GalleryImageFilename);
    auto targetSize = GalleryImageFilename.size();
    std::cout << "Target Size = " << targetSize << std::endl;

    if(targetSize == 0) ::exit(0);

    std::vector<int64_t> GalleryIndex(targetSize);
    for( size_t i = 0; i < targetSize; ++i )
    {
        //register face into facedatabase
        std::string &filename = GalleryImageFilename[i];
        int64_t &index = GalleryIndex[i];
	std::cerr << "Registering... " << filename << std::endl;
        seeta::cv::ImageData image = cv::imread( filename );
        auto id = engine.Register( image );
        index = id;
        std::cerr << "Registered id = " << id << std::endl;
    }
    std::map<int64_t, std::string> GalleryIndexMap;
    for( size_t i = 0; i < GalleryIndex.size(); ++i )
    {
        // save index and name pair
        if( GalleryIndex[i] < 0 ) continue;
        GalleryIndexMap.insert( std::make_pair( GalleryIndex[i], GalleryImageFilename[i] ) );
    }

    using namespace std::chrono;
    steady_clock::time_point start,end;
    std::vector<std::string> peopleList;

    std::cout << "----query people directory: "<< people_dir << std::endl;
    parse_image_list(people_dir,peopleList);
    std::cout << "People Size = " << peopleList.size() << std::endl;
    for(auto& filename: peopleList)
    {
        seeta::cv::ImageData image = cv::imread(filename);

        // Detect all faces
	start = steady_clock::now();
        std::vector<SeetaFaceInfo> faces = engine.DetectFaces( image );
	end = steady_clock::now();
	std::cout << std::endl << "Detect " << filename << " Faces cost: " << duration_cast<milliseconds>(end-start).count() << " ms " << std::endl;

        for( SeetaFaceInfo &face : faces )
        {
            // Query top 1
            int64_t index = -1;
            float similarity = 0;

	    start = steady_clock::now();
	    auto points = engine.DetectPoints(image, face);
	    end = steady_clock::now();
	    std::cout << "    DetectPoints cost: " << duration_cast<milliseconds>(end-start).count() << " ms " << std::endl;

	    start = steady_clock::now();
            auto queried = engine.QueryTop( image, points.data(), 1, &index, &similarity );
	    end = steady_clock::now();
	    std::cout << "    QueryTop cost: " << duration_cast<milliseconds>(end-start).count() << " ms " << std::endl;

	    std::cout << "    Detect Face : [" << face.pos.x << "," << face.pos.y << "," << face.pos.width << "," << face.pos.height << "]" << std::endl;
            //cv::rectangle( frame, cv::Rect( face.pos.x, face.pos.y, face.pos.width, face.pos.height ), CV_RGB( 128, 128, 255 ), 3 );

	    // no face queried from database
	    if (queried < 1) continue;

	    std::cout << "    QueryTop1 :" << queried << ",index = " << index << ",similarity = " << similarity << std::endl;

            // similarity greater than threshold, means recognized
            if( similarity > threshold )
            {
		std::cout << "    People [" << filename << "] Recognized (>" << threshold << ") As Name : " << GalleryIndexMap[index] << "!" << std::endl << std::endl;
                //cv::putText( frame, GalleryIndexMap[index], cv::Point( face.pos.x, face.pos.y - 5 ), 3, 1, CV_RGB( 255, 128, 128 ) );
            }
        }
    }
}

