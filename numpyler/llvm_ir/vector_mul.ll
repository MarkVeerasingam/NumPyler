; Multiply vector by an undefined array

%memref = type { i8*, i8*, i64, [1 x i64], [1 x i64] }

define void @vector_mul(%memref* %arg, i32 %scalar) {
entry:
  %shape_ptr = getelementptr inbounds %memref, %memref* %arg, i32 0, i32 3, i32 0
  %size = load i64, i64* %shape_ptr
  %data_ptr = getelementptr inbounds %memref, %memref* %arg, i32 0, i32 1
  %aligned_ptr = load i8*, i8** %data_ptr
  %int_ptr = bitcast i8* %aligned_ptr to i32*

  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %loop ]
  %index_ptr = getelementptr i32, i32* %int_ptr, i64 %i
  %val = load i32, i32* %index_ptr
  %mul_val = mul i32 %val, %scalar
  store i32 %mul_val, i32* %index_ptr
  %i_next = add i64 %i, 1
  %cond = icmp slt i64 %i_next, %size
  br i1 %cond, label %loop, label %done

done:
  ret void
}