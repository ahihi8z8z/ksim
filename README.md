# KSIM 
Gymnasium + [Faas-sim](https://github.com/edgerun/faas-sim)

## Cấu trúc code
- ksim_env/envs/ksim.py: môi trường gymnasium
- ksim_env/utils/ : Môi trường serverless, dựa trên framework faas-sim
    - Phần nào viết mới/chỉnh sửa thì sẽ sẽ có tiền tố `K`, ví dụ: `KBenchmark` là viết lại [Benchmark](https://github.com/edgerun/faas-sim/blob/master/sim/benchmark.py) 
- exec_time.csv, invocation.csv: thời gian thực thi và số lần gọi hàm, lấy từ [dataset azure](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)
- train_rl.py, eval_rl.py, baseline.py: các kịch bản thí nghiệm.
## Cài đặt
Code chạy trên python 3.8.10, các version khác chưa test.
Trong thư mục ksim, chạy các lệnh sau:

```{shell}
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/edgerun/faas-sim.git
cd faas-sim/
python3 -m pip install .
cd ..
python3 -m pip install -r requirements.txt .
```

## Chạy 
Có 3 file:
- train_rl.py: train model RL dùng [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/)
- eval_rl.py: dùng model RL sau khi train
- baseline.py: chạy các thuật toán scaler heuristic khác để 

Chạy lệnh:
```{shell}
python3 {ten_file}
```
Kết quả chạy lưu vào `logs/{name_prefix}` tương ứng. Với trường hợp train RL thì quá trình train được log vào `ksim_tensorboard`. Để xem quá trình này, mở terminal khác và chạy lệnh
```{shell}
tensorboard --logdir ksim_tensorboard/ --bind_all
```
Sau đó truy cập `http://{host ip}:6006/` để xem

## Các thay đổi chính so với faas-sim default
- Sửa trạng thái của replica (xem AppState)
    - `NULL`: replica không tồn tại
    - `CONCEIVED`: replica đã được thêm vào control plane nhưng chưa được tạo trên node
    - `STARTING`: khởi tạo replica trên node
    - `SUSPENDED`: replica bị tạm dừng trước khi bị scale down
    - `UNLOADED_MODEL`: replica đã được tạo trên node nhưng chưa khởi tạo model AI
    - `LOADING_MODEL`: đang khởi tạo model AI
    - `UNLOADING_MODEL`: đang gỡ bỏ model AI
    - `LOADED_MODEL`: đã khởi tạo model AI trên replica, sẵn sàng phục vụ yêu cầu
    - `ACTIVING`: replica đang phục vụ yêu cầu
- Thêm các API đi kèm để môi trường gym có thể đổi trạng thái replica (KSystem)
- Custom logic mô phỏng hoạt động replica theo trạng thái mới (xem KFunctionSimulator)
- Benchmark chạy theo từng step để môi trường gym có thể điều khiển được.
- Sinh request theo azure dataset (Xem KBenchmark) 

## Hoạt động của hệ thống:
- 1 replica/container/deployment/service. 
- Theo chu kì, môi trường gym gọi API để đổi trạng thái replica trong môi trường faas sim. Nếu nhảy hơn một trạng thái (ví dụ chuyển 10 replica từ NULL sang LOADED_MODEL), API sẽ nhảy từng trạng thái (chuyển 10 replica từ NULL sang UNLOADED_MODEL, sau đó chuyển 10 replica từ UNLOADED_MODEL sang LOADED_MODEL). Xem comment để biết vấn đề với cách làm này.
- Các hàm chuyển trạng thái:
    - `scale_up`: tạo replica mới, ứng với chuyển từ `NULL` sang `UNLOADED_MODEL`. Lưu ý là hàm này chạy xong thì yêu cầu tạo replica được đẩy vào scheduler và xếp hàng đợi. Tuy nhiên hiện tại đang để thời gian schedule = 0 nên replica vẫn sẽ được tạo ngay lập tức. 
    - `do_load_model`: chuyển replica ở trạng thái `UNLOADED_MODEL` sang `LOADED_MODEL`
    - `do_unload_model`: chuyển replica ở `LOADED_MODEL` sang `UNLOADED_MODEL`.
    - `scale down`: xóa các replica ở trạng thái `UNLOADED_MODEL`. Bị giới hạn bởi cấu hình `scale_min` (xem phần cấu hình).
- Vị trí đặt replica mới dựa theo [scheduler mặc định của faas-sim](https://github.com/edgerun/skippy-core/blob/754b20b0d5a3ee597d17682ef555ea1bf1340ea5/skippy/core/scheduler.py).
- Khi traffic được đẩy vào (faas.invoke()), hệ thống tìm các replica ở trạng thái LOADED_MODEL hoặc ACTIVING để chuyển tiếp. Nếu không có replica nào ở 2 trạng thái trên thì hệ thống polling để tìm mỗi 0.5s, tối đa 37s. Quá 37s không có thì request bị drop. Nếu có nhiều hơn một replica ở các trạng thái trên, load balancer reqest theo round robin (xem KRoundRobinLoadBalancer).
- Khi request đến replica, nếu replica đang bận phục vụ reqest khác, nó được xếp vào queue và đợi vô hạn cho tới khi đến lượt được phục vụ. Mỗi replica chỉ phục vụ một request tại một thời điểm, có thể sửa hành vi này qua cấu hình (xem phần cấu hình). 

## Cấu hình
Vì môi trường gym chứa môi trường faas-sim, toàn bộ cấu hình của hai môi trường được truyền vào khi khởi tạo môi trường gym. Hiện tại bao gồm các cấu hình sau:
- `num_servers`: số lượng server. Mỗi server là một skippy node của faas-sim với 88 CPU và 188 GB RAM.
- `server_cap`: dung lượng server, có 2 giá trị là RAM và CPU. 
- `random_init`: Nếu false, faas-sim sẽ khởi tạo với `scale_min` replica. Ngược lại, khởi tạo ngẫu nhin một số lượng replica. Mặc định là false.
- `timeout`: Hết timeout thì môi trường gym và môi trường faas-sim sẽ dừng. Tính theo thời gian của môi trường faas-sim, đơn vị là giây, mặc định là 1209600 giây (14 ngày).
- `max_episode_steps`: Hết `max_episode_steps` thì môi trường gym sẽ dừng. Tính theo số bước của môi trường gym.
- `step_size`: mỗi bước trong môi trường gym ứng với bao nhiêu giây trong môi trường faas-sim.
- `scaler_config`: chọn thuật toán auto scale. Các thuật toán chạy độc lập và tính theo service, có nghĩa là một service có thể dùng nhiều bộ auto scaler cùng lúc. Lưu ý khi train và eval RL thì phải để tất cả là false (true cũng được, nếu muốn kết hợp RL+ heuristic).
    - `scale_by_requests`: thuật toán auto scale replica theo số request đến hệ thống.
    - `scale_by_average_requests_per_replica`: bản cải tiến của `scale_by_requests`, tính đến số request trung bình mỗi replica phải xử lý.
    - `scale_by_queue_requests_per_replica`: auto scale theo số request hiện nằm trong queue của từng replica. Lưu ý thuật toán này chưa test.
- `services`: cấu hình của từng service, key là `HashFunction` lấy trong azure dataset. Lưu ý hiện tại một số đoạn code trong môi trường gym mặc định là chỉ có 1 serivice nên chỗ này chỉ nên nạp 1 service.
    - `req_profile_file`: đường dẫn đến file csv chứa số lần gọi hàm của service tương ứng. File này bao gồm 4 + n cột. 4 cột đầu là metadata, n cột sau ứng với số lời gọi hàm trung bình mỗi phút. File này nên chỉ có 1 dòng duy nhất ứng với một service (nhiều dòng cũng được nhưng chưa test).
    - `exec_time_file`: đường dẫn đến file csv chứa đặc trưng thống kê của thời gian thực thi của service tương ứng. File này bao gồm 7 hàng, mỗi hàng là 1 ngày theo thứ tự trong dataset azure.
    - `sim_duration`: môi trường faas-sim sẽ lấy ngẫu nhiên một đoạn dài `sim_duration` trong dataset azure để mô phỏng traffic, ví dụ là 24 thì lấy ngẫu nhiên 1 ngày trong 14 ngày. Đơn vị là giờ, nên để chia hết cho 24. Nếu muốn hành vi xác định, để là 336 giờ (ứng với 14 ngày, lúc này môi trường faas-sim sẽ luôn bắt đầu từ ngày 1).
    - `state_resource_usage`: Tài nguyên replica tiêu thụ tại từng trạng thái. Ví dụ `UNLOADED_MODEL` tiêu thụ 5 cpu thì một node có 10 replica ở `UNLOADED_MODEL` sẽ tiêu thụ 50 cpu. Chuyển sang `LOADED_MODEL` tiêu thụ 10 cpu thì sau khi chuyển node bị chiếm 100 cpu.  Riêng với `ACTIVING`, thì tài nguyên ở đây đại diện cho mỗi request đang được phục vụ. Lưu ý với các trạng thái có thể điều khiển, thời gian cần để là 0. Đơn vị thời gian là giây, đơn vị cpu là milisecond (chia 1000 ra số core), ~~đơn vị RAM là byte~~ giờ RAM truyền vào dạng string theo đơn vị thông thường (K, M, G, Ki, Mi, Gi). 
    - `image_size`: kích thước image của service. Đơn vị là byte.
    - `resources`: Tài nguyên container yêu cầu. Bao gồm RAM và CPU.
    - `num_workers`: số lượng request xử lý song song tại mỗi replica. Mặc định lả 1.
    - `scale_min`: số lượng replica `LOADED_MODEL` tối thiểu trong hệ thống.
    - `scale_max`: số lượng replica khác `NULL` tối đa trong hệ thống.
    - `rps_threshold`: tham số của thuật toán auto scale `scale_by_requests`. 
    - `rps_threshold_duration`: tham số của thuật toán auto scale `scale_by_requests`. 
    - `alert_window`: tham số của thuật toán auto scale `scale_by_average_requests_per_replica`.
    - `target_average_rps`: tham số của thuật toán auto scale `scale_by_average_requests_per_replica`.
    - `target_queue_length`: tham số của thuật toán auto scale `scale_by_queue_requests_per_replica`.


## Hướng dẫn train RL
- Các tham số ảnh hưởng đến thời gian chạy:
    - `n_envs`: số môi trường chạy song song, lý thuyết thì mỗi môi trường 1 core nhưng càng nhiều thì chạy càng lâu.
    - `n_eval_episodes`: số episode mỗi lần evaluation
    - `eval_freq`: bao nhiêu step thì evaluation 1 lần
    - `total_timesteps`: tổng cộng train bao nhiêu step.
- Với cấu hình hiện tại thì, chạy song song `8` môi trường. Với mỗi môi trường chạy được `72` steps thì evaluation `1` episode. Train tổng `228*14*30` step tương đương với mỗi môi trường train trong `14` tháng. 
- Muốn train nhanh thì nên giảm các tham số trên, ưu tiên `total_timesteps`.
- Muốn đổi traffic pattern thì đổi đường dẫn đến file trong config. Format file traffic thì xem mô tả trong phần cấu hình bên trên.