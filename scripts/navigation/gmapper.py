import numpy as np
import cv2 

class Position:
    def __init__(self, x:float, y:float, z:float, free:bool=True, valid:bool=False, obstacle:bool=False) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.free = free
        self.valid = valid
        self.obstacle = obstacle
        

class GMapping:
    def __init__(self, x_size:int=20, y_size:int=20, resolution:float=0.1) -> None:
        self.x_size = int(x_size / resolution)
        self.y_size = int(y_size / resolution)
        self.resolution = resolution
        self.tolerance = self.resolution / 2
        
        self.size = (x_size * y_size) / resolution**2
        self.positions = [[Position(x * resolution , y * resolution, 0) for x in range(self.x_size)] for y in range(self.y_size)]
        
        
    def get_position(self, x: float, y: float) -> Position:
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return self.positions[grid_y][grid_x]
    

    def update_position(self, x: float, y: float, z:float, free:bool, valid:bool, obstacle:bool) -> None:
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        self.positions[grid_y][grid_x].z = z
        self.positions[grid_y][grid_x].free = free
        self.positions[grid_y][grid_x].valid = valid
        self.positions[grid_y][grid_x].obstacle = obstacle
        
        
    def is_valid_position(self, x:float, y:float):
        return (x % self.resolution < self.tolerance) and (y % self.resolution < self.tolerance)
    
    
    def get_valid_positions(self):
        return [obj for riga in self.positions for obj in riga if obj.valid]
    
    
    def find_obstacle(self, cur_pos:np.ndarray, depth: np.ndarray, threshold:float=0.5, debug:bool=False) -> list:
        height, width = depth.shape
        
        w_pow = 10**int(np.floor(np.log10(width / 3)))
        h_pow = 10**int(np.floor(np.log10(height / 3)))
        w_size = np.floor((width / 3) / w_pow) * w_pow
        h_size = np.floor((height / 3) / h_pow) * h_pow
        w_center = width // 2 
        
        crops = np.array([
            [0, 0, w_size, h_size],
            [width - w_size, 0 , width, h_size],
            [w_center - (w_size // 2), 0, w_center + (w_size // 2), h_size]
        ]).astype(np.int32)
        crops = crops[np.argsort(crops[:, 0])[::-1]]
        
        front_patch = cur_pos[:2, 3] + (cur_pos[:3, :3] @ (np.array([[1], [0], [0]]))).reshape(-1)[:2]
        front_patches = np.array([
            front_patch + np.array([0, -self.resolution]),
            front_patch,
            front_patch + np.array([0, self.resolution])
        ])
        
        obstacles = []
        for i, crop in enumerate(crops):
            cropped_img = depth[crop[1]:crop[3], crop[0]:crop[2]]
            if not np.isnan(depth).all() and np.nanmin(cropped_img) < threshold:
                position = self.get_position(front_patches[i][0], front_patches[i][0])
                self.update_position(position.x, position.y, np.nanmin(cropped_img), free=False, valid=False, obstacle=True)
                obstacles.append(front_patches[i])
            
            if debug:    
                cv2.imshow("crop", cropped_img)
                cv2.waitKey(0)
        
        if debug:
            for crop in crops:
                cv2.rectangle(depth, tuple(crop[:2].tolist()), tuple(crop[2:].tolist()), 0.5, 2)
            cv2.imshow("depth", depth)
            cv2.waitKey(0)
        
        return obstacles
    