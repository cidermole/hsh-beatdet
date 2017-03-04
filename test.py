from hsh_signal.app_parser import AppData
import matplotlib.pyplot as plt
from hsh_beatdet import beatdet_getrr_v1, beatdet_getrr_v2, beatdet
import random
from os.path import basename


if __name__ == '__main__':
    measurements = AppData.list_measurements()

    #for ad in measurements:
    for i in range(100):
        ad = random.choice(measurements)
        mf = basename(ad.meta_filename)
        print mf

        #ppg = ad.ppg_parse_beatdetect('getrr')
        data = ad.ppg_data()

        rr1 = beatdet(data, beatdet_getrr_v1)
        rr2 = beatdet(data, beatdet_getrr_v2)

        rr1.plot(c='b')

        rr1.scatter(color='r', s=30)
        rr2.scatter(color='g', s=30)

        plt.title(mf)
        plt.show()
