using MAT

# Open .fig file.
fig = matopen("toys/homoclinics/shilnikov-hopf/diagram_als.fig")

# Read *all* data.
data = read(fig)

# Store graphics data.
graphics = data["hgM_070000"]["GraphicsObjects"]