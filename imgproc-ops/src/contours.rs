use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
};
use std::collections::VecDeque;

// ============================================================================
// Data Structures
// ============================================================================

/// A point in 2D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    /// X coordinate
    pub x: i32,
    /// Y coordinate
    pub y: i32,
}

impl Point {
    /// Create a new point
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A rectangle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// X coordinate of top-left corner
    pub x: i32,
    /// Y coordinate of top-left corner
    pub y: i32,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Calculate area
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }
}

/// Contour hierarchy types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContourHierarchy {
    /// External (outermost) contour
    External,
    /// Hole (inner boundary)
    Hole,
    /// Nested contour with parent and children indices
    Nested {
        /// Parent contour index
        parent: Option<usize>,
        /// Child contour indices
        children: Vec<usize>,
    },
}

/// A contour (closed curve)
#[derive(Debug, Clone)]
pub struct Contour {
    /// Points defining the contour
    pub points: Vec<Point>,
    /// Hierarchy information
    pub hierarchy: ContourHierarchy,
}

impl Contour {
    /// Create a new contour
    #[must_use]
    pub fn new(points: Vec<Point>, hierarchy: ContourHierarchy) -> Self {
        Self { points, hierarchy }
    }

    /// Get number of points
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if contour is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Contour retrieval mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourRetrievalMode {
    /// Only external (outermost) contours
    External,
    /// All contours without hierarchy
    List,
    /// Full hierarchy tree
    Tree,
}

/// Contour approximation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximation {
    /// Store all contour points
    None,
    /// Compress horizontal, vertical, and diagonal segments
    Simple,
}

// ============================================================================
// Contour Detection
// ============================================================================

/// Find contours in binary image
///
/// Uses a border-following algorithm to detect contours in binary images.
/// The image must be binary (0 = background, non-zero = foreground).
///
/// # References
/// - Suzuki & Abe (1985). "Topological Structural Analysis of Digitized Binary Images"
#[derive(Debug, Clone)]
pub struct FindContours {
    /// Contour retrieval mode
    pub mode: ContourRetrievalMode,
    /// Approximation method
    pub method: ContourApproximation,
}

impl FindContours {
    /// Create a new contour finder
    #[must_use]
    pub const fn new(mode: ContourRetrievalMode, method: ContourApproximation) -> Self {
        Self { mode, method }
    }

    /// Find contours in binary image
    ///
    /// # Arguments
    /// * `binary_image` - Binary input image (0 = background, >0 = foreground)
    ///
    /// # Returns
    /// * Vector of detected contours
    pub fn find_contours(&self, binary_image: &Image<Gray8>) -> Result<Vec<Contour>> {
        let width = binary_image.width() as i32;
        let height = binary_image.height() as i32;

        // Create labels image for tracking visited pixels
        let mut labels = vec![vec![0u8; width as usize]; height as usize];

        let mut contours = Vec::new();

        // Scan image for contour starting points
        for y in 0..height {
            for x in 0..width {
                let pixel = binary_image.get_pixel(x as u32, y as u32)?.value;

                // Check if this is a foreground pixel that hasn't been visited
                if pixel > 0 && labels[y as usize][x as usize] == 0 {
                    // Found a new contour
                    let contour_points = self.trace_contour(
                        binary_image,
                        &mut labels,
                        Point::new(x, y),
                        width,
                        height,
                    )?;

                    if !contour_points.is_empty() {
                        let hierarchy = match self.mode {
                            ContourRetrievalMode::External | ContourRetrievalMode::List => {
                                ContourHierarchy::External
                            }
                            ContourRetrievalMode::Tree => ContourHierarchy::External,
                        };

                        contours.push(Contour::new(contour_points, hierarchy));
                    }
                }
            }
        }

        Ok(contours)
    }

    /// Trace a single contour using border following
    fn trace_contour(
        &self,
        image: &Image<Gray8>,
        labels: &mut [Vec<u8>],
        start: Point,
        width: i32,
        height: i32,
    ) -> Result<Vec<Point>> {
        let mut points = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        labels[start.y as usize][start.x as usize] = 1;

        // 8-connectivity offsets
        const OFFSETS: [(i32, i32); 8] = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];

        while let Some(pt) = queue.pop_front() {
            points.push(pt);

            // Check 8 neighbors
            for &(dx, dy) in &OFFSETS {
                let nx = pt.x + dx;
                let ny = pt.y + dy;

                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    if labels[ny as usize][nx as usize] == 0 {
                        let pixel = image.get_pixel(nx as u32, ny as u32)?.value;
                        if pixel > 0 {
                            labels[ny as usize][nx as usize] = 1;
                            queue.push_back(Point::new(nx, ny));
                        }
                    }
                }
            }
        }

        // Apply simplification if requested
        if matches!(self.method, ContourApproximation::Simple) {
            Ok(Self::simplify_contour(&points))
        } else {
            Ok(points)
        }
    }

    /// Simplify contour by removing redundant points
    fn simplify_contour(points: &[Point]) -> Vec<Point> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        let mut simplified = Vec::new();
        simplified.push(points[0]);

        for i in 1..points.len() - 1 {
            let prev = points[i - 1];
            let curr = points[i];
            let next = points[i + 1];

            // Keep point if it's not collinear with neighbors
            if !Self::is_collinear(prev, curr, next) {
                simplified.push(curr);
            }
        }

        simplified.push(points[points.len() - 1]);
        simplified
    }

    /// Check if three points are collinear
    fn is_collinear(p1: Point, p2: Point, p3: Point) -> bool {
        let dx1 = p2.x - p1.x;
        let dy1 = p2.y - p1.y;
        let dx2 = p3.x - p2.x;
        let dy2 = p3.y - p2.y;

        // Cross product should be zero for collinear points
        (dx1 * dy2 - dy1 * dx2).abs() <= 1
    }
}

// ============================================================================
// Contour Features
// ============================================================================

/// Image moments for shape analysis
#[derive(Debug, Clone, Copy)]
pub struct Moments {
    /// Spatial moment m00
    pub m00: f64,
    /// Spatial moment m10
    pub m10: f64,
    /// Spatial moment m01
    pub m01: f64,
    /// Spatial moment m20
    pub m20: f64,
    /// Spatial moment m11
    pub m11: f64,
    /// Spatial moment m02
    pub m02: f64,
    /// Central moment mu20
    pub mu20: f64,
    /// Central moment mu11
    pub mu11: f64,
    /// Central moment mu02
    pub mu02: f64,
}

/// Compute contour features and descriptors
pub struct ContourFeatures;

impl ContourFeatures {
    /// Calculate contour area using Green's theorem (shoelace formula)
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Contour area in pixels
    #[must_use]
    pub fn area(contour: &Contour) -> f64 {
        if contour.points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = contour.points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let p1 = contour.points[i];
            let p2 = contour.points[j];

            area += (p1.x * p2.y) as f64;
            area -= (p2.x * p1.y) as f64;
        }

        (area / 2.0).abs()
    }

    /// Calculate contour perimeter
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Contour perimeter in pixels
    #[must_use]
    pub fn perimeter(contour: &Contour) -> f64 {
        if contour.points.len() < 2 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        let n = contour.points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += contour.points[i].distance_to(&contour.points[j]);
        }

        perimeter
    }

    /// Calculate bounding box
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Axis-aligned bounding rectangle
    #[must_use]
    pub fn bounding_box(contour: &Contour) -> Rect {
        if contour.points.is_empty() {
            return Rect::new(0, 0, 0, 0);
        }

        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;

        for pt in &contour.points {
            min_x = min_x.min(pt.x);
            max_x = max_x.max(pt.x);
            min_y = min_y.min(pt.y);
            max_y = max_y.max(pt.y);
        }

        Rect::new(
            min_x,
            min_y,
            (max_x - min_x + 1) as u32,
            (max_y - min_y + 1) as u32,
        )
    }

    /// Calculate minimum enclosing circle
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * (center, radius) of minimum enclosing circle
    #[must_use]
    pub fn min_enclosing_circle(contour: &Contour) -> (Point, f64) {
        if contour.points.is_empty() {
            return (Point::new(0, 0), 0.0);
        }

        // Simplified: use bounding box center and maximum distance
        let bbox = Self::bounding_box(contour);
        let center_x = bbox.x + bbox.width as i32 / 2;
        let center_y = bbox.y + bbox.height as i32 / 2;
        let center = Point::new(center_x, center_y);

        let mut max_dist: f64 = 0.0;
        for pt in &contour.points {
            let dist = center.distance_to(pt);
            max_dist = max_dist.max(dist);
        }

        (center, max_dist)
    }

    /// Compute convex hull using Jarvis march (gift wrapping) algorithm
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Convex hull as vector of points
    #[must_use]
    pub fn convex_hull(contour: &Contour) -> Vec<Point> {
        if contour.points.len() < 3 {
            return contour.points.clone();
        }

        let mut hull = Vec::new();
        let points = &contour.points;

        // Find leftmost point
        let mut leftmost_idx = 0;
        for (i, pt) in points.iter().enumerate() {
            if pt.x < points[leftmost_idx].x ||
               (pt.x == points[leftmost_idx].x && pt.y < points[leftmost_idx].y) {
                leftmost_idx = i;
            }
        }

        let mut current_idx = leftmost_idx;

        loop {
            hull.push(points[current_idx]);
            let mut next_idx = (current_idx + 1) % points.len();

            // Find the most counterclockwise point
            for i in 0..points.len() {
                if Self::cross_product(
                    points[current_idx],
                    points[i],
                    points[next_idx],
                ) > 0 {
                    next_idx = i;
                }
            }

            current_idx = next_idx;

            if current_idx == leftmost_idx {
                break;
            }

            if hull.len() > points.len() {
                break; // Safety check
            }
        }

        hull
    }

    /// Cross product for orientation test
    fn cross_product(o: Point, a: Point, b: Point) -> i32 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Compute moments
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Image moments
    #[must_use]
    pub fn moments(contour: &Contour) -> Moments {
        let mut m00 = 0.0;
        let mut m10 = 0.0;
        let mut m01 = 0.0;
        let mut m20 = 0.0;
        let mut m11 = 0.0;
        let mut m02 = 0.0;

        // Compute spatial moments
        for pt in &contour.points {
            let x = pt.x as f64;
            let y = pt.y as f64;

            m00 += 1.0;
            m10 += x;
            m01 += y;
            m20 += x * x;
            m11 += x * y;
            m02 += y * y;
        }

        // Compute centroids
        let cx = if m00 > 0.0 { m10 / m00 } else { 0.0 };
        let cy = if m00 > 0.0 { m01 / m00 } else { 0.0 };

        // Compute central moments
        let mut mu20 = 0.0;
        let mut mu11 = 0.0;
        let mut mu02 = 0.0;

        for pt in &contour.points {
            let x = pt.x as f64 - cx;
            let y = pt.y as f64 - cy;

            mu20 += x * x;
            mu11 += x * y;
            mu02 += y * y;
        }

        Moments {
            m00,
            m10,
            m01,
            m20,
            m11,
            m02,
            mu20,
            mu11,
            mu02,
        }
    }

    /// Calculate center of mass from moments
    ///
    /// # Arguments
    /// * `moments` - Image moments
    ///
    /// # Returns
    /// * Center of mass as Point
    #[must_use]
    pub fn center_of_mass(moments: &Moments) -> Point {
        if moments.m00 > 0.0 {
            Point::new(
                (moments.m10 / moments.m00).round() as i32,
                (moments.m01 / moments.m00).round() as i32,
            )
        } else {
            Point::new(0, 0)
        }
    }

    /// Calculate circularity (compactness)
    ///
    /// Circularity = 4π × area / perimeter²
    /// Perfect circle = 1.0, less circular shapes < 1.0
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Circularity value (0-1)
    #[must_use]
    pub fn circularity(contour: &Contour) -> f64 {
        let area = Self::area(contour);
        let perimeter = Self::perimeter(contour);

        if perimeter > 0.0 {
            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
        } else {
            0.0
        }
    }

    /// Calculate aspect ratio
    ///
    /// Aspect ratio = width / height of bounding box
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Aspect ratio
    #[must_use]
    pub fn aspect_ratio(contour: &Contour) -> f64 {
        let bbox = Self::bounding_box(contour);
        if bbox.height > 0 {
            bbox.width as f64 / bbox.height as f64
        } else {
            0.0
        }
    }

    /// Calculate solidity
    ///
    /// Solidity = contour_area / convex_hull_area
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Solidity value (0-1)
    #[must_use]
    pub fn solidity(contour: &Contour) -> f64 {
        let area = Self::area(contour);
        let hull = Self::convex_hull(contour);
        let hull_contour = Contour::new(hull, ContourHierarchy::External);
        let hull_area = Self::area(&hull_contour);

        if hull_area > 0.0 {
            area / hull_area
        } else {
            0.0
        }
    }

    /// Calculate extent
    ///
    /// Extent = contour_area / bounding_box_area
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Extent value (0-1)
    #[must_use]
    pub fn extent(contour: &Contour) -> f64 {
        let area = Self::area(contour);
        let bbox = Self::bounding_box(contour);
        let bbox_area = bbox.area() as f64;

        if bbox_area > 0.0 {
            area / bbox_area
        } else {
            0.0
        }
    }
}

// ============================================================================
// Douglas-Peucker Contour Approximation
// ============================================================================

/// Douglas-Peucker contour approximation
///
/// Reduces the number of points in a contour while preserving its shape.
///
/// # References
/// - Douglas & Peucker (1973). "Algorithms for the Reduction of the Number of Points"
#[derive(Debug, Clone)]
pub struct ApproxPolyDP {
    /// Approximation accuracy (maximum distance from original contour)
    pub epsilon: f64,
}

impl ApproxPolyDP {
    /// Create a new approximation operation
    ///
    /// # Arguments
    /// * `epsilon` - Maximum distance between original curve and approximation
    ///
    /// # Returns
    /// * Approximation operation
    pub fn new(epsilon: f64) -> Result<Self> {
        if epsilon < 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon".to_string(),
                message: "Epsilon must be non-negative".to_string(),
            });
        }

        Ok(Self { epsilon })
    }

    /// Approximate contour using Douglas-Peucker algorithm
    ///
    /// # Arguments
    /// * `contour` - Input contour
    ///
    /// # Returns
    /// * Simplified point list
    pub fn approximate(&self, contour: &Contour) -> Vec<Point> {
        if contour.points.len() < 3 {
            return contour.points.clone();
        }

        let mut result = Vec::new();
        self.douglas_peucker(&contour.points, 0, contour.points.len() - 1, &mut result);
        result
    }

    /// Recursive Douglas-Peucker implementation
    fn douglas_peucker(&self, points: &[Point], start: usize, end: usize, result: &mut Vec<Point>) {
        let mut max_dist = 0.0;
        let mut max_idx = start;

        // Find point with maximum distance from line segment
        for i in start + 1..end {
            let dist = Self::perpendicular_distance(points[i], points[start], points[end]);
            if dist > max_dist {
                max_dist = dist;
                max_idx = i;
            }
        }

        // If max distance is greater than epsilon, recursively simplify
        if max_dist > self.epsilon {
            self.douglas_peucker(points, start, max_idx, result);
            result.push(points[max_idx]);
            self.douglas_peucker(points, max_idx, end, result);
        } else if result.is_empty() || result[result.len() - 1] != points[start] {
            result.push(points[start]);
        }
    }

    /// Calculate perpendicular distance from point to line segment
    fn perpendicular_distance(pt: Point, line_start: Point, line_end: Point) -> f64 {
        let dx = (line_end.x - line_start.x) as f64;
        let dy = (line_end.y - line_start.y) as f64;
        let length_sq = dx * dx + dy * dy;

        if length_sq == 0.0 {
            return pt.distance_to(&line_start);
        }

        let t = ((pt.x - line_start.x) as f64 * dx + (pt.y - line_start.y) as f64 * dy) / length_sq;
        let t = t.clamp(0.0, 1.0);

        let proj_x = line_start.x as f64 + t * dx;
        let proj_y = line_start.y as f64 + t * dy;

        let dist_x = pt.x as f64 - proj_x;
        let dist_y = pt.y as f64 - proj_y;

        (dist_x * dist_x + dist_y * dist_y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0, 0);
        let p2 = Point::new(3, 4);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_rect_area() {
        let rect = Rect::new(0, 0, 10, 20);
        assert_eq!(rect.area(), 200);
    }

    #[test]
    fn test_contour_features() {
        let points = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];
        let contour = Contour::new(points, ContourHierarchy::External);

        let area = ContourFeatures::area(&contour);
        assert!(area > 90.0 && area < 110.0); // Allow for floating point errors

        let perimeter = ContourFeatures::perimeter(&contour);
        assert!(perimeter > 38.0 && perimeter < 42.0);

        let bbox = ContourFeatures::bounding_box(&contour);
        assert_eq!(bbox.width, 11);
        assert_eq!(bbox.height, 11);
    }
}
