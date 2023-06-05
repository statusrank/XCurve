# sh scripts/convergence_one.sh config_train/iNaturalist/convergence/soprc_v4_sgd.yaml 4 &
# sh scripts/convergence_one.sh config_train/iNaturalist/convergence/triplet.yaml 5 &
# sh scripts/convergence_one.sh config_train/iNaturalist/convergence/smooth_ap.yaml 6 &
# sh scripts/convergence_one.sh config_train/iNaturalist/convergence/fast_ap.yaml 3 &
# sh scripts/convergence_one.sh config_train/iNaturalist/convergence/black_box.yaml 3 &

sh scripts/convergence_one.sh config_train/SOP/soprc_v4_sgd.yaml 4 &
sh scripts/convergence_one.sh config_train/SOP/triplet.yaml 5 &
sh scripts/convergence_one.sh config_train/SOP/smooth_ap.yaml 6 &
sh scripts/convergence_one.sh config_train/SOP/fast_ap.yaml 3 &
sh scripts/convergence_one.sh config_train/SOP/black_box.yaml 3 &

# sh scripts/convergence_one.sh config_train/VehID/soprc_v4_sgd.yaml 4 &
# sh scripts/convergence_one.sh config_train/VehID/triplet.yaml 5 &
# sh scripts/convergence_one.sh config_train/VehID/smooth_ap.yaml 6 &
# sh scripts/convergence_one.sh config_train/VehID/fast_ap.yaml 3 &
# sh scripts/convergence_one.sh config_train/VehID/black_box.yaml 3 &

# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_bs56.yaml 4 &
# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_bs112.yaml 5 &
# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_bs168.yaml 6 &

# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_t0.01.yaml 4 &
# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_t0.05.yaml 5 &
# sh scripts/convergence_one.sh config_train/iNaturalist/ablations/soprc_v4_sgd_t0.2.yaml 6 &
