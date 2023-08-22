import pymap3d as pm


class CarMovement:
    def __init__(self, gps_imu_0, timestamp_0):
        self.gps_imu_0 = gps_imu_0
        self.timestamp_0 = timestamp_0
    
    def get_lat_lon_alt(self, gps_imu):
        lat, lon, h = gps_imu.lat, gps_imu.lon, gps_imu.alt
        return lat, lon, h
    
    def calc_movement(self, gps_imu):
        lat, lon, h = self.get_lat_lon_alt(gps_imu)
        lat0, lon0, h0 = self.get_lat_lon_alt(self.gps_imu_0)
        return pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
    
    def calc_time(self, timestamp):
        return (timestamp - self.timestamp_0).total_seconds()
    