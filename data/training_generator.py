bpm = 100
bar_length = 4
bar_duration = 60.0 / (bpm / float(bar_length))
full_bars = 4

barline_filename = "training_barlines.txt"
with open(barline_filename, 'w') as barline_file:
    for i in xrange(full_bars + 3):
        barline_file.write("%f\tbl\n" % (1.0 + (i * bar_duration)))

gem_filename = "training_gems.txt"
with open(gem_filename, 'w') as gem_file:
    # Write silence gems
    bar_start = 1.0
    gem_file.write("%f\t255\n" % (bar_start))
    gem_file.write("%f\t255\n" % (bar_start + (bar_duration / 4.0)))
    gem_file.write("%f\t255\n" % (bar_start + (2.0 * bar_duration / 4.0)))
    gem_file.write("%f\t255\n" % (bar_start + (3.0 * bar_duration / 4.0)))

    for i in xrange(full_bars + 1):
        # Write actual gems
        bar_start = 1.0 + ((i + 1) * bar_duration)
        gem_file.write("%f\t1\n" % (bar_start))
        gem_file.write("%f\t2\n" % (bar_start + (bar_duration / 4.0)))
        gem_file.write("%f\t3\n" % (bar_start + (2.0 * bar_duration / 4.0)))
        gem_file.write("%f\t2\n" % (bar_start + (3.0 * bar_duration / 4.0)))
