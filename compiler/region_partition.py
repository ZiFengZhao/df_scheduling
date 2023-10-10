import configparser
import logging
import re


class RegionPartitioner:
    def __init__(self, config_file='conf.ini', verbose=False, energy_costs=None):

        self.energy_cost = energy_costs
        self.config_file = config_file

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        self.logger = logging.getLogger('{}.{}'.format(__name__, 'RegionPartitioner'))
        self.logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        self.logger.debug("Creating RegionPartitioner Object")

        # System
        # off-chip访存通道位于df_arch两侧，记录这些通道坐标
        mem_channels_pos = self.config.get('system', 'mem_channels_pos')
        # 使用正则表达式匹配坐标字符串，并提取坐标值
        coordinates = re.findall(r'\((\d+),(\d+)\)', mem_channels_pos)

        # 将提取的坐标值转换为二元元组
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        self.mem_channels_pos = coordinates
        self.logger.debug("Relative position of Off-chip Memory Channels： {}".format(self.mem_channels_pos))

        # Tile Array
        self.df_arch_dim = [self.config.getint('tile_array', 'row'),
                            self.config.getint('tile_array', 'col')]
        self.logger.debug("Dataflow architecture dimensions: {}".format(self.df_arch_dim))

        # Tile
        tile_sram = {}

        tile_sram['act'] = self.config.getint('tile', 'Act_SRAM')
        self.logger.debug("Activation SRAM size for tile: {:,} Bytes".format(tile_sram['act']))

        tile_sram['wgt'] = self.config.getint('tile', 'Wgt_SRAM')
        self.logger.debug("Weight SRAM size for tile: {:,} Bytes".format(tile_sram['wgt']))

        self.tile_sram = tile_sram

    def collect_regions(self):

        height = self.df_arch_dim[0]
        width = self.df_arch_dim[1]
        tile_sram_size = sum((self.tile_sram['act'], self.tile_sram['wgt']))
        regions_lst = []

        # 循环遍历所有可能的左上角坐标
        for x1 in range(width):
            for y1 in range(height):
                # 循环遍历所有可能的右下角坐标
                for x2 in range(x1, width):
                    for y2 in range(y1, height):
                        region_width = x2 - x1 + 1
                        region_height = y2 - y1 + 1

                        # 检查region的面积是否大于0
                        assert region_width > 0 and region_height > 0

                        # 检查region是否有至少一个off-chip访存通道
                        region_pos = (x1, y1, x2, y2)
                        if self.has_mem_channels(region_pos):
                            num_tiles = region_width * region_height
                            region_sram = num_tiles * tile_sram_size
                            region = (region_pos, num_tiles, region_sram)
                            regions_lst.append(region)

        return regions_lst

    def has_mem_channels(self, region_pos):

        has_flg = False

        x1, y1, x2, y2 = region_pos
        tile_array_height = self.df_arch_dim[0]
        tile_array_width = self.df_arch_dim[1]

        assert 0 <= x1 <= x2 < tile_array_width
        assert 0 <= y1 <= y2 < tile_array_height

        if x1 > 0 and x2 < (tile_array_width - 1):
            has_flg = False
        elif x1 == 0: # 贴近左边的region
            for chan_x, chan_y in self.mem_channels_pos:
                if chan_x == 0 and y1 <= chan_y <= y2:
                    has_flg = True
                    break
        else:
            assert x2 == (tile_array_width - 1)
            for chan_x, chan_y in self.mem_channels_pos:
                if chan_x == (tile_array_width - 1) and y1 <= chan_y <= y2:
                    has_flg = True
                    break

        return has_flg


if __name__ == '__main__':
    config_file = '../configs/hw_cfg.ini'
    verbose = True

    # Creating a RegionPartitioner object
    reg_partitioner = RegionPartitioner(config_file=config_file, verbose=verbose)

    avail_region_lst = reg_partitioner.collect_regions()
    for pos, num_tiles, sram_size in avail_region_lst:
        print("Region position:({},{},{},{})".format(pos[0], pos[1], pos[2], pos[3]))
        print("Region contains {} tiles".format(num_tiles))
        print("SRAM size in the region:{}".format(sram_size))

    print("Totally {} available regions.".format(len(avail_region_lst)))
