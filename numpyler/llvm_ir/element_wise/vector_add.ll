%memref = type { i8*, i8*, i64, [1 x i64], [1 x i64] }

define void @vector_add(%memref* %a, %memref* %b, %memref* %out) {
entry:
  %size_ptr = getelementptr inbounds %memref, %memref* %a, i32 0, i32 3, i32 0
  %size = load i64, i64* %size_ptr

  %a_data_ptr_ptr = getelementptr inbounds %memref, %memref* %a, i32 0, i32 1
  %a_aligned = load i8*, i8** %a_data_ptr_ptr
  %a_int_ptr = bitcast i8* %a_aligned to i32*

  %b_data_ptr_ptr = getelementptr inbounds %memref, %memref* %b, i32 0, i32 1
  %b_aligned = load i8*, i8** %b_data_ptr_ptr
  %b_int_ptr = bitcast i8* %b_aligned to i32*

  %out_data_ptr_ptr = getelementptr inbounds %memref, %memref* %out, i32 0, i32 1
  %out_aligned = load i8*, i8** %out_data_ptr_ptr
  %out_int_ptr = bitcast i8* %out_aligned to i32*

  br label %loop

loop:
  %i = phi i64 [0, %entry], [%i_next, %loop]

  %a_idx_ptr = getelementptr i32, i32* %a_int_ptr, i64 %i
  %b_idx_ptr = getelementptr i32, i32* %b_int_ptr, i64 %i
  %out_idx_ptr = getelementptr i32, i32* %out_int_ptr, i64 %i

  %a_val = load i32, i32* %a_idx_ptr
  %b_val = load i32, i32* %b_idx_ptr
  %add_val = add i32 %a_val, %b_val

  store i32 %add_val, i32* %out_idx_ptr

  %i_next = add i64 %i, 1
  %cond = icmp slt i64 %i_next, %size
  br i1 %cond, label %loop, label %done

done:
  ret void
}
