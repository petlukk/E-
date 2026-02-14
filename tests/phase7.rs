mod common;

#[cfg(test)]
mod tests {
    use crate::common::{assert_c_interop, assert_output, assert_shared_lib_interop};

    #[test]
    fn test_struct_field_access() {
        assert_output(
            r#"
struct Point {
    x: f64,
    y: f64,
}

func main() {
    let p: Point = Point { x: 3.5, y: 7.25 }
    println(p.x)
    println(p.y)
}
"#,
            "3.5\n7.25",
        );
    }

    #[test]
    fn test_struct_field_assign() {
        assert_output(
            r#"
struct Counter {
    value: i32,
}

func main() {
    let mut c: Counter = Counter { value: 0 }
    c.value = 42
    println(c.value)
}
"#,
            "42",
        );
    }

    #[test]
    fn test_struct_multiple_fields() {
        assert_output(
            r#"
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

func main() {
    let r: Rect = Rect { x: 1.0, y: 2.0, w: 10.0, h: 20.0 }
    println(r.w)
    println(r.h)
}
"#,
            "10\n20",
        );
    }

    #[test]
    fn test_struct_pass_to_function() {
        assert_output(
            r#"
struct Vec2 {
    x: f64,
    y: f64,
}

func length_sq(v: Vec2) -> f64 {
    return v.x * v.x + v.y * v.y
}

func main() {
    let v: Vec2 = Vec2 { x: 3.0, y: 4.0 }
    println(length_sq(v))
}
"#,
            "25",
        );
    }

    #[test]
    fn test_struct_return_from_function() {
        assert_output(
            r#"
struct Pair {
    a: i32,
    b: i32,
}

func make_pair(x: i32, y: i32) -> Pair {
    return Pair { a: x, b: y }
}

func main() {
    let p: Pair = make_pair(10, 20)
    println(p.a)
    println(p.b)
}
"#,
            "10\n20",
        );
    }

    #[test]
    fn test_struct_pointer_field_access_c_interop() {
        assert_c_interop(
            r#"
struct Particle {
    x: f32,
    y: f32,
    mass: f32,
}

export func get_x(p: *Particle) -> f32 {
    return p.x
}

export func get_mass(p: *Particle) -> f32 {
    return p.mass
}
"#,
            r#"
#include <stdio.h>

typedef struct { float x; float y; float mass; } Particle;
extern float get_x(const Particle*);
extern float get_mass(const Particle*);

int main() {
    Particle p = { 1.5f, 2.5f, 10.0f };
    printf("%g %g\n", get_x(&p), get_mass(&p));
    return 0;
}
"#,
            "1.5 10",
        );
    }

    #[test]
    fn test_struct_mut_pointer_field_assign_c_interop() {
        assert_c_interop(
            r#"
struct Point {
    x: f32,
    y: f32,
}

export func set_point(p: *mut Point, nx: f32, ny: f32) {
    p.x = nx
    p.y = ny
}
"#,
            r#"
#include <stdio.h>

typedef struct { float x; float y; } Point;
extern void set_point(Point*, float, float);

int main() {
    Point p = { 0.0f, 0.0f };
    set_point(&p, 3.14f, 2.71f);
    printf("%g %g\n", p.x, p.y);
    return 0;
}
"#,
            "3.14 2.71",
        );
    }

    #[test]
    fn test_struct_array_c_interop() {
        assert_c_interop(
            r#"
struct Vec2 {
    x: f32,
    y: f32,
}

export func sum_x(vecs: *Vec2, n: i32) -> f32 {
    let mut total: f32 = 0.0
    let mut i: i32 = 0
    while i < n {
        total = total + vecs[i].x
        i = i + 1
    }
    return total
}
"#,
            r#"
#include <stdio.h>

typedef struct { float x; float y; } Vec2;
extern float sum_x(const Vec2*, int);

int main() {
    Vec2 vecs[] = { {1.0f, 10.0f}, {2.0f, 20.0f}, {3.0f, 30.0f} };
    printf("%g\n", sum_x(vecs, 3));
    return 0;
}
"#,
            "6",
        );
    }

    #[test]
    fn test_struct_init_particles_c_interop() {
        assert_c_interop(
            r#"
struct Particle {
    x: f32,
    y: f32,
    mass: f32,
}

export func init_particles(p: *mut Particle, n: i32) {
    let mut i: i32 = 0
    while i < n {
        p[i].x = 0.0
        p[i].y = 0.0
        p[i].mass = 1.0
        i = i + 1
    }
}
"#,
            r#"
#include <stdio.h>

typedef struct { float x; float y; float mass; } Particle;
extern void init_particles(Particle*, int);

int main() {
    Particle ps[3] = { {99,99,99}, {99,99,99}, {99,99,99} };
    init_particles(ps, 3);
    printf("%g %g %g\n", ps[0].x, ps[0].y, ps[0].mass);
    printf("%g %g %g\n", ps[2].x, ps[2].y, ps[2].mass);
    return 0;
}
"#,
            "0 0 1\n0 0 1",
        );
    }

    #[test]
    fn test_struct_with_i32_fields() {
        assert_output(
            r#"
struct Score {
    player: i32,
    points: i32,
}

func main() {
    let s: Score = Score { player: 1, points: 100 }
    println(s.player)
    println(s.points)
}
"#,
            "1\n100",
        );
    }

    #[test]
    fn test_shared_lib_output() {
        assert_shared_lib_interop(
            r#"
export func add(a: i32, b: i32) -> i32 {
    return a + b
}
"#,
            r#"
#include <stdio.h>
extern int add(int, int);
int main() {
    printf("%d\n", add(17, 25));
    return 0;
}
"#,
            "42",
        );
    }

    #[test]
    fn test_shared_lib_struct_export() {
        assert_shared_lib_interop(
            r#"
struct Point {
    x: f32,
    y: f32,
}

export func get_x(p: *Point) -> f32 {
    return p.x
}

export func set_y(p: *mut Point, val: f32) {
    p.y = val
}
"#,
            r#"
#include <stdio.h>

typedef struct { float x; float y; } Point;
extern float get_x(const Point*);
extern void set_y(Point*, float);

int main() {
    Point p = { 5.0f, 0.0f };
    set_y(&p, 9.5f);
    printf("%g %g\n", get_x(&p), p.y);
    return 0;
}
"#,
            "5 9.5",
        );
    }

    #[test]
    fn test_struct_field_in_expression() {
        assert_output(
            r#"
struct Vec2 {
    x: f64,
    y: f64,
}

func main() {
    let a: Vec2 = Vec2 { x: 3.0, y: 4.0 }
    let len_sq: f64 = a.x * a.x + a.y * a.y
    println(len_sq)
}
"#,
            "25",
        );
    }
}
