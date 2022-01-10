#include "iostream"
#include <opencv2/core/core.hpp>
#include "Detection.h"

int main(int argc, char* argv[])
{
	// I read the folders in "data"
	for (int j = 1; j < 5; j++)
	{
		char fn[20];
		sprintf_s(fn, "..\\data\\dataset%d", j);
		std::vector<cv::String> img;
		int img_obj[10];
		int img_scene[10];

		std::string path = std::string(fn);
		cv::glob(path, img);
		int num_obj = 0;
		int num_scene = 0;

		// I read the images
		for (int k = 0; k < img.size(); k++)
		{
			std::size_t found_scene = img[k].find("scene");
			std::size_t found_obj = img[k].find("obj");
			// Until the end of the string
			if (found_scene != std::string::npos)
			{
				img_scene[num_scene] = k;
				num_scene++;
			}
			if (img_scene == 0)
			{
				std::cout << "There is no scene with the correct name" << std::endl;
			}
			// Until the end of the string
			else if (found_obj != std::string::npos)
			{
				img_obj[num_obj] = k;
				num_obj++;
			}
			if (img_obj == 0)
			{
				std::cout << "There is no object with the correct name" << std::endl;
			}
		}

		// I search in the scenes the objects
		for (int i = 0; i < num_obj; i++)
		{
			for (int k = 0; k < num_scene; k++)
			{

				std::cout << "Load image " << img[img_obj[i]] << " in scene " << img[img_scene[k]] << std::endl;
				// Creation of the object 
				ObjectDetection od = ObjectDetection(img[img_obj[i]], img[img_scene[k]]);

				// Detect key points in the image
				od.detectObj();

				// Detect object in the scene
				od.detectScene();

				// Find matches between the images
				od.match();

				// Print the result image
				od.ShowResults();
			}
		}
	}
	return 0;
}
