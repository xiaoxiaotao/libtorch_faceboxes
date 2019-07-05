#include <torch/script.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include<algorithm>

using namespace std;

int kIMAGE_SIZE=1024;
int kCHANNELS = 3;

void softmax(vector<vector<float>> input,vector<vector<float>> &output){
    vector<float> temp;
    for(int i=0;i<input.size();i++){
        //cout<<i<<endl;
        temp.push_back(exp(input[i][0])/(exp(input[i][0])+exp(input[i][1])));
        temp.push_back(exp(input[i][1])/(exp(input[i][0])+exp(input[i][1])));
        output.push_back(temp);
        temp.clear();
    }
}
void read_txt(vector<float> &data_set,string file){
    ifstream f;
    f.open(file,ios::in);
    float tmp;
    for (int i = 0; i < 21824*4 ; i++)
    {

      f >> tmp;
      data_set.push_back(tmp);
    }


    f.close();
}
void frame_pad(cv::Mat input,cv::Mat &output){
    int w=input.cols;
    int h=input.rows;
    //cout<<w<<" "<<h<<endl;
    int borderType = cv::BORDER_CONSTANT;
    int top;
    int bottom;
    int left;
    int right;
    if(w>=h){
        top =int((w-h)/2);
        bottom = int((w-h)/2);
        left = 0;
        right = 0;

    }else{
        top =0;
        bottom = 0;
        left = int((h-w)/2);
        right = int((h-w)/2);
    }
    //Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    copyMakeBorder(input, output, top, bottom, left, right, borderType);
}
typedef struct Bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float area;

}Bbox;
bool cmpScore(Bbox lsh, Bbox rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}
void nms(vector<Bbox> &boundingBox_, const float overlap_threshold){

    if(boundingBox_.empty()){
        return;
    }
    //对各个候选框根据score的大小进行升序排列
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    vector<int> vPick;
    int nPick = 0;
    multimap<float, int> vScores;   //存放升序排列后的score和对应的序号
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i){
        vScores.insert(pair<float, int>(boundingBox_[i].score, i));
    }
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;  //反向迭代器，获得vScores序列的最后那个序列号
        vPick[nPick] = last;
        nPick += 1;
        for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //转换成了两个边界框相交区域的边长
            maxX = ((minX-maxX)>0)? (minX-maxX) : 0;
            maxY = ((minY-maxY)>0)? (minY-maxY) : 0;

            IOU = (maxX * maxY)/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - maxX * maxY);

            if(IOU > overlap_threshold){
                it = vScores.erase(it);    //删除交并比大于阈值的候选框,erase返回删除元素的下一个元素
            }else{
                it++;
            }
        }
    }

    vPick.resize(nPick);
    vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}
int main(int argc, const char *argv[]) {


  std::string modelpath="/home/tao/libtorch/example/faceboxes_pytorch_c++/pytorch/faceboxes_v3.pt";
  std::shared_ptr<torch::jit::script::Module> module =torch::jit::load(modelpath);
  std::cout << "== Switch to GPU mode" << std::endl;
   //to GPU
  module->to(at::kCUDA);
  assert(module != nullptr);
  //std::string file_name = "/home/tao/libtorch/example/faceboxes_pytorch_c++/pytorch/picture/img_68.jpg";

  //cv::Mat image=cv::imread(file_name);
  //cv::resize(image,image,cv::Size(1024,1024));

  //torch::jit::Stack outputs = model->forward({input}).toTuple()->elements();
  //std::cout<<output2<<std::endl;
  float loc_preds_array[21824][4];
  float conf_preds_array[21824][2];
  string file="/home/tao/libtorch/example/faceboxes_pytorch_c++/pytorch/data.txt";
  vector<float> boxes_v1;
  vector<vector<float>> boxes_v2;
  read_txt(boxes_v1,file);
  float boxes_array[21824][4];
  float variances[2]={0.1,0.2};
  vector<vector<float>> conf;//
  vector<vector<float>> loc;//
  vector<vector<float>> conf_output;//after softmax
  vector<int> idx;
  vector<float> conf_temp;
  vector<float> loc_temp;
  float boxes_decode[21824][4];
  float boxes_decode_[21824][4];
  Bbox bbox;
  vector<Bbox> nms_data;
  vector<float> temp_boxes;
  for(int i=0;i<21824;i++){
      temp_boxes.push_back(boxes_v1[i*4+0]);
      temp_boxes.push_back(boxes_v1[i*4+1]);
      temp_boxes.push_back(boxes_v1[i*4+2]);
      temp_boxes.push_back(boxes_v1[i*4+3]);
      boxes_v2.push_back(temp_boxes);
      temp_boxes.clear();
  }
  for (int i =0 ;i<21824;i++){
      for(int j=0;j<4;j++){
          boxes_array[i][j] = boxes_v2[i][j];

      }
  }

  cv::VideoCapture cap;
  cv::Mat image;
  while(true){
      cap.read(image);
      if(image.empty())
      {
          cout<<"cant open the camera"<<endl;
          //cap.open("rtsp://admin:admin888@192.168.1.33:554/h264/ch1/main/av_stream");
          cap.open(0);
          usleep(50*1000);
      continue;
      }
      else{
          frame_pad(image,image);
          cv::resize(image,image,cv::Size(1024,1024));
          image.convertTo(image, CV_32F, 1.0 / 255.0);
          auto input_tensor = torch::from_blob(image.data, {1,kIMAGE_SIZE,kIMAGE_SIZE,kCHANNELS});
          input_tensor = input_tensor.permute({0, 3, 1, 2});
          input_tensor = input_tensor.to(at::kCUDA);
          //auto out_tensor = module->forward({input_tensor}).toTensor();
          torch::jit::Stack outputs = module->forward({input_tensor}).toTuple()->elements();
          //outputs = outputs.to(at::kCPU);
          auto output1=outputs[0].toTensor().to(at::kCPU);
          auto output2=outputs[1].toTensor().to(at::kCPU);
          auto loc_outputss=output1.accessor<float, 3>();
          auto conf_outputss=output2.accessor<float, 3>();


          for(int i=0;i<21824;i++){
              loc_preds_array[i][0]=loc_outputss[0][i][0];
              loc_preds_array[i][1]=loc_outputss[0][i][1];
              loc_preds_array[i][2]=loc_outputss[0][i][2];
              loc_preds_array[i][3]=loc_outputss[0][i][3];
              conf_preds_array[i][0]=conf_outputss[0][i][0];
              conf_preds_array[i][1]=conf_outputss[0][i][1];

          }
          for(int i=0;i<21824;i++){
              boxes_decode[i][0] = loc_preds_array[i][0] * variances[0] * boxes_array[i][2] + boxes_array[i][0];
              boxes_decode[i][1] = loc_preds_array[i][1] * variances[0] * boxes_array[i][3] + boxes_array[i][1];
              boxes_decode[i][2] = exp(loc_preds_array[i][2] * variances[1]) * boxes_array[i][2];
              boxes_decode[i][3] = exp(loc_preds_array[i][3] * variances[1]) * boxes_array[i][3];

              boxes_decode_[i][0] = boxes_decode[i][0] - boxes_decode[i][2]/2.;
              boxes_decode_[i][1] = boxes_decode[i][1] - boxes_decode[i][3]/2.;
              boxes_decode_[i][2] = boxes_decode[i][0] + boxes_decode[i][2]/2.;
              boxes_decode_[i][3] = boxes_decode[i][1] + boxes_decode[i][3]/2.;

              if(conf_preds_array[i][1]>conf_preds_array[i][0]){
                  idx.push_back(i);

              }
          }
          //cout<<idx.size()<<endl;
          for(int i=0;i<idx.size();i++){
              int id=idx[i];
              loc_temp.push_back(boxes_decode_[id][0]);
              loc_temp.push_back(boxes_decode_[id][1]);
              loc_temp.push_back(boxes_decode_[id][2]);
              loc_temp.push_back(boxes_decode_[id][3]);
              loc.push_back(loc_temp);
              conf_temp.push_back(conf_preds_array[id][0]);
              conf_temp.push_back(conf_preds_array[id][1]);
              conf.push_back(conf_temp);
              loc_temp.clear();
              conf_temp.clear();
          }
          softmax(conf,conf_output);//compute confidence
          for(int i=0;i<idx.size();i++)
          {
              bbox.x1=float(loc[i][0]);
              bbox.y1=float(loc[i][1]);
              bbox.x2=float(loc[i][2]);
              bbox.y2=float(loc[i][3]);
              bbox.score=conf_output[i][1];
              bbox.area=(bbox.x2-bbox.x1)*(bbox.y2-bbox.y1);
              nms_data.push_back(bbox);
          }
          nms(nms_data,0.35);
          for(int i=0;i<nms_data.size();i++){

              if(nms_data[i].score>0.8){

               int x1=int(nms_data[i].x1*1024);
               int y1=int(nms_data[i].y1*1024);
               int x2=int(nms_data[i].x2*1024);
               int y2=int(nms_data[i].y2*1024);
              cv::rectangle(image,cv::Rect(x1,y1,x2-x1,y2-y1),cv::Scalar(0,255,255),3);
              }
          }
          idx.clear();
          loc.clear();
          conf_output.clear();
          conf.clear();
          nms_data.clear();
          cv::resize(image,image,cv::Size(500,500));
          cv::imshow("result",image);
          cv::waitKey(2);
      }
  }

}




