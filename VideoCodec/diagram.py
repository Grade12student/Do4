import matplotlib.pyplot as plt

# Bitrates and PSNRs for H.264
bitrates_264 = [0.5918819059, 0.2493294946, 0.1174311343, 0.06531616512]
psnrs_264 = [0.9761527889, 0.9667849138, 0.9557047309, 0.9396002221]

bitrates_265 = [0.6408461806, 0.2577161265, 0.1153772377, 0.06096880787]
psnrs_265 = [0.9791658642, 0.970395697, 0.9611346854, 0.9490031292]

my_bitrates = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
my_psnrs = [0.94666318, 0.95921518, 0.96812108, 0.97383252]



plt.figure(figsize=(6, 6))

# Use 'r--' for a red dashed line
plt.plot(bitrates_264, psnrs_264, 'r*--', label='H.264')
plt.plot(bitrates_265, psnrs_265, 'b*--', label='H.265')
plt.plot(my_bitrates, my_psnrs, 'k*-', label='Our Codec')

plt.legend(loc="lower right")
plt.xlabel('Bitrate (bpp)')
plt.ylabel('MS-SSIM')
#plt.title('Rate-Distortion Curve')
plt.gca().set_xlim(right=0.8)
plt.gca().set_ylim(bottom=0.93)

plt.tight_layout()
plt.savefig('rd_curve.pdf')
plt.show()

