* Các thư viện sử dụng

* Python 3
* Tensorflow GPU = 1.4.1 (không sử dụng TF thường)
* numpy
* tqdm
* sklearn
* rouge
* pyrouge

Chuẩn bị cho mô hình:

* Link tải data và model 
Đăng nhập bằng mail TDT và ghi đè vào thư mục gốc(bắt buộc):

https://drive.google.com/file/d/1ObzrJNzDAmudH5A7CgtxfBvXQGXHNeq7/view?usp=sharing


*Cài đặt pearl (phục vụ việc tạo vocabulary ban đầu trước khi train, phải tạo lại từ đầu nếu đổi máy) (bắt buộc):

https://www.perl.org/get.html


* Cài đặt CUDA (Nhớ chọn phiên bản phù hợp vs tf-gpu) (bắt buộc):

https://developer.nvidia.com/cuda-10.1-download-archive-base


*Cài đặt PYROUGE (Đọc kỹ phần trả lời, có thể sẽ mất 30-60 phút để cài đặt):

https://stackoverflow.com/questions/47045436/how-to-install-the-python-package-pyrouge-on-microsoft-windows

____Huấn luyện toàn bộ mô hình____

    + Huấn luyện từ đầu mô hình:

    python main.py --mode=train --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --pretrain_dis_data_path=./data/discriminator_train_data.npz --restore_best_model=False

    + Phục hồi quá trình huấn luyện:

    python main.py --mode=train --data_path=./data/train.bin --vocab_path=./data/vocab --log_root=./log --pretrain_dis_data_path=./data/discriminator_train_data.npz --restore_best_model=True

___Decode____: 
	+ test toàn bộ tập test 11k tập)

    python main.py --mode=decode --data_path=./data/test.bin --vocab_path=./data/vocab --log_root=./log --single_pass=True

	+ test Ngẫu nhiên mỗi lần một ví dụ

    python main.py --mode=decode --data_path=./data/test.bin --vocab_path=./data/vocab --log_root=./log --single_pass=False
	
