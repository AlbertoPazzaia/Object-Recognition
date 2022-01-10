#ifndef _Detection_H_
#define _Detection_

//#include <stdio.h>
//#include <stdlib.h> 
#include "iostream"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

class ObjectDetection
{
private:
	// Parameters for detector
	//cv::Ptr<cv::SIFT> detector; // Uncomment to use SIFT?
	cv::Ptr<cv::ORB> detector;

	std::vector<cv::KeyPoint> keypoints_obj;
	std::vector<cv::KeyPoint> keypoints_scene;

	cv::Mat descriptors_obj;
	cv::Mat descriptors_scene;

	// Parameters for matcher
	cv::Ptr<cv::BFMatcher> matcher;

	std::vector<std::vector< cv::DMatch>>  matches;
	std::vector< cv::DMatch > good_matches;
	
	// Points for homography
	std::vector<cv::Point2f> objPoints; 
	std::vector<cv::Point2f> scenePoints;

public:
	// We initializate the images for the class
	cv::Mat image_obj;
	cv::Mat image_scene;

	// Gray images to fnd the keypoints
	cv::Mat image_gray_scene;
	cv::Mat image_gray_obj;

	// Homography and result image
	cv::Mat homography;
	cv::Mat result_img;

	ObjectDetection(std::string object, std::string scene)
	{
		// Reading and conversion of scene and object images
		image_obj = cv::imread(object);
		if (image_obj.empty())
		{
			std::cout << "Impossible to read the object image" << std::endl;
			return;
		}
		cv::cvtColor( image_obj, image_gray_obj, cv::COLOR_BGR2GRAY);

		image_scene = cv::imread(scene);
		if (image_scene.empty())
		{
			std::cout << "Impossible to read the scene image" << std::endl;
			return;
		}
		cv::cvtColor( image_scene, image_gray_scene, cv::COLOR_BGR2GRAY);


		// Parameters for ORB
		// Max number of features to extract!!!
		int nfeatures = 1500;

		// Scale step between different pyramid levels
		float scaleFactor = 1.2f;

		// Numbers of pyramid levels
		int nlevels = 8;

		// Avoid computing features close to edges(set to 21)
		int edgeThreshold = 21;
		int firstLevel = 0;

		// 2 for comparison between couples of points
		int WTA_K = 2;

		// Size of patch for feature computation
		int patchSize = 21;

		//Threshold in FAST algorithm
		int fastThreshold = 20;


		// Creation of detector and matcher for every image
		// int nOctaveLayers = 3; // Uncomment to use SIFT?
		// double contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6; // Uncomment to use SIFT?
		// detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma); // Uncomment to use SIFT?
		detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE,
			patchSize, fastThreshold);
		// Use NORM_L2 for SIFT
		matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
	}

	void detectObj()
	{
		// Detect Keypoints and descriptors in the object images
		detector->detectAndCompute(image_gray_obj, cv::Mat(), keypoints_obj, descriptors_obj, false);

		// To see the Keypoints in every object loaded
		/*
		cv::Mat results_obj;
		cv::drawKeypoints(image_obj, keypoints_obj, results_obj);
		cv::namedWindow("Object keypoints",cv::WINDOW_AUTOSIZE);
    	cv::imshow("Object keypoints", results_obj);
    	cv::waitKey(0);
    	 */
	}

	void detectScene()
	{
		// Detect Keypoints and descriptors in the scene images
		detector->detectAndCompute(image_gray_scene, cv::Mat(), keypoints_scene, descriptors_scene, false);

		// To see the Keypoints in every object loaded
		/*
		cv::Mat results_scene;
		cv::drawKeypoints(image_scene, keypoints_scene, results_scene);
		cv::namedWindow("Scene keypoints",cv::WINDOW_NORMAL);
    	cv::imshow("Scene keypoints", results_scene);
    	cv::waitKey(0);
		*/
	}

	void match()
	{
		int temp = 2;
		bool nope = false;
		cv::Mat empty;
		matcher->cv::DescriptorMatcher::knnMatch(descriptors_scene, descriptors_obj, matches, temp, empty, nope);
	}

	int ShowResults()
	{
		// Creation ofthe result image
		result_img = image_scene.clone();

		const float ratio_thresh = 0.75f;

		int min = 1000;
		int max = 0;

		for (size_t i = 0; i < matches.size(); ++i)
		{
			if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);  //objPoints

				if (min > matches[i][0].distance)
					min = matches[i][0].distance;
				if (max < matches[i][0].distance)
					max = matches[i][0].distance;
			}
		}

			if (min > 20 || good_matches.empty())
		{
			cv::namedWindow("Object image", cv::WINDOW_NORMAL);
			cv::imshow("Object image", image_obj);
			cv::namedWindow("Result image", cv::WINDOW_NORMAL);
			cv::imshow("Result image", result_img);
			cv::waitKey(0);
			std::cout << "There'aren't good matches" << std::endl;
			return 1;
		}

		// Select the good matches
		for( size_t i = 0; i < good_matches.size(); i++ )
		{
			// Get the keypoints from the good matches
			objPoints.push_back( keypoints_obj[good_matches[i].trainIdx ].pt );
			scenePoints.push_back( keypoints_scene[good_matches[i].queryIdx].pt );
		}

		if (good_matches.size()<30)
		{
			cv::namedWindow("Object image", cv::WINDOW_NORMAL);
			cv::imshow("Object image", image_obj);
			cv::namedWindow("Result image", cv::WINDOW_NORMAL);
			cv::imshow("Result image", result_img);
			cv::waitKey(0);
			std::cout << "There'aren't enough good matches" << std::endl;
			return 1;
		}

		// Threshold on reprojection error for a point to be considered inlier
		double ransacReprojThreshold = 3;

		const int maxIters = 2000;
		const double confidence = 0.995;
		cv::Mat empty;

		// Homography
		homography = cv::findHomography(objPoints, scenePoints, cv::RANSAC, ransacReprojThreshold, empty, maxIters, confidence);

		// I set the corners for the object image
		std::vector<cv::Point2f> obj_corners(4);
		obj_corners[0] = cv::Point(0,0);
		obj_corners[1] = cv::Point(image_gray_obj.cols, 0);
		obj_corners[2] = cv::Point(image_gray_obj.cols, image_gray_obj.rows);
		obj_corners[3] = cv::Point(0, image_gray_obj.rows );

		// I set the corners for the scene image
		std::vector<cv::Point2f> scene_corners(4);

		if (homography.empty())
		{
			cv::namedWindow("Object image", cv::WINDOW_NORMAL);
			cv::imshow("Object image", image_obj);
			cv::namedWindow("Result image", cv::WINDOW_NORMAL);
			cv::imshow("Result image", result_img);
			cv::waitKey(0);
			std::cout << "Impossible to persevere the homography" << std::endl;
			return 1;
		}

		// Transforms the corners of the objects in corners in the scene using the matrix resulting by the homography
		cv::perspectiveTransform(obj_corners, scene_corners, homography);

		// I draw a green line on the scene
		cv::line(result_img, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4, cv::LINE_8, 0);
		cv::line(result_img, scene_corners[1], scene_corners[2], cv::Scalar(0, 255, 0), 4, cv::LINE_8, 0);
		cv::line(result_img, scene_corners[2], scene_corners[3], cv::Scalar(0, 255, 0), 4, cv::LINE_8, 0);
		cv::line(result_img, scene_corners[3], scene_corners[0], cv::Scalar(0, 255, 0), 4, cv::LINE_8, 0);

		cv::namedWindow("Object image",cv::WINDOW_NORMAL);
    	cv::imshow("Object image", image_obj);
		cv::namedWindow("Result image",cv::WINDOW_NORMAL);
    	cv::imshow("Result image", result_img);
    	cv::waitKey(0);
		return 0;
	}
};
#endif
