import numpy as np
from scipy.stats import truncnorm

def generate_city_distance_matrix(size=128):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            # Sinh số ngẫu nhiên theo phân phối chuẩn, giới hạn trong [0, 1]
            distance = truncnorm.rvs((0 - 0.5) / 0.2, (1 - 0.5) / 0.2, loc=0.5, scale=0.2)
            distance = np.round(distance, 1)  # Làm tròn 1 chữ số thập phân
            distance = np.clip(distance, 0.0, 1.0)  # Đảm bảo nằm trong [0, 1]
            matrix[i, j] = distance
            matrix[j, i] = distance  # Ma trận đối xứng
    return matrix

if __name__ == "__main__":
    # Tạo ma trận
    city_matrix = generate_city_distance_matrix()
    
    # In ra để kiểm tra (tùy chọn)
    print(city_matrix)
    
    # Lưu vào file
    output_path = r"C:\Code\IT3020_Toanroirac\MFEA\GLS\Gentest\city_matrix.txt"
    np.savetxt(output_path, city_matrix, fmt="%.1f")  # Lưu với 1 chữ số thập phân
    print(f"Ma trận đã được lưu vào: {output_path}")